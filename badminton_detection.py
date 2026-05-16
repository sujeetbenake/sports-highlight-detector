"""
运动帧影 Alano · 羽毛球专项检测引擎
基于 Sports-Highlight-Detector 接口规范设计
融合 vision_pipeline.py 的运动理解规则引擎
"""

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

class BadmintonDetector:
    """
    羽毛球专项检测器 —— 不依赖字幕，纯视觉理解
    
    检测能力：
    ✅ 杀球检测（手臂速度 + 球速突增）
    ✅ 多拍回合检测（球往返计数）
    ✅ 得分检测（球落地 + 球员停动）
    ✅ 鱼跃救球检测（身体快速下移）
    """
    
    def __init__(self, confidence_threshold=0.3):
        self.sport_type = 'badminton'
        self.confidence_threshold = confidence_threshold
        self.model = YOLO('yolov8n.pt')
        
        # 羽毛球检测参数
        self.min_players = 2          # 至少2名球员
        self.action_threshold = 0.3   # 动作触发阈值
        
        # 追踪历史
        self.track_history = deque(maxlen=60)  # 60帧历史
        self.ball_positions = deque(maxlen=30) # 球的位置历史
        self.player_positions = {}             # 球员位置追踪
        self.rally_count = 0                   # 多拍计数
        self.last_ball_side = None             # 球在场地哪一侧
        
        # 杀球检测
        self.arm_speeds = deque(maxlen=10)
        self.ball_speeds = deque(maxlen=10)
        
        # 动作持续性 (与足球检测对齐)
        self.action_history = deque(maxlen=10)
        self.action_persistence = 3
        
    def detect_action(self, frame, debug=False):
        """
        主检测接口 —— 与 SportDetector.detect_action() 完全兼容
        
        Returns:
            dict: {
                'players': [...],
                'ball': {...} or None,
                'has_action': bool,
                'action_type': str or None,
                'highlight_score': float (0-10)
            }
        """
        # 运行 YOLOv8 检测
        results = self.model(frame, conf=self.confidence_threshold, classes=[0, 32, 43])
        
        detections = {
            'players': [],
            'ball': None,
            'rackets': [],
            'has_action': False,
            'action_type': None,
            'highlight_score': 0.0,
        }
        
        if len(results) == 0:
            return detections
        
        result = results[0]
        current_ball_center = None
        
        # ---- 解析检测结果 ----
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, bbox)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            if cls_id == 0 and conf >= 0.3:  # Person → 球员
                detections['players'].append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'center': center,
                    'bbox_height': y2 - y1,
                    'bbox_width': x2 - x1,
                })
            elif cls_id == 32 and conf >= 0.15:  # Sports ball → 羽毛球
                detections['ball'] = {
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'center': center,
                }
                current_ball_center = center
            elif cls_id == 43 and conf >= 0.2:  # Tennis racket → 球拍
                detections['rackets'].append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'center': center,
                })
        
        # ---- 执行规则引擎 ----
        if len(detections['players']) >= self.min_players:
            score = 0
            
            # 规则1: 球速突增 → 杀球检测
            ball_speed = self._calc_ball_speed(current_ball_center)
            if ball_speed > 50:
                score += 3
                detections['action_type'] = 'smash'
            
            # 规则2: 手臂动作剧烈 → 杀球/扣杀
            arm_speed = self._calc_arm_speed(detections['players'])
            if arm_speed > 30:
                score += 2
                if detections['action_type'] is None:
                    detections['action_type'] = 'vigorous_action'
            
            # 规则3: 多拍回合
            self._update_rally_count(current_ball_center)
            if self.rally_count >= 6:
                score += self.rally_count * 0.5  # 多拍加分
                if score > 4:
                    detections['action_type'] = 'long_rally'
            
            # 规则4: 球员快速下移 → 鱼跃救球
            dive_score = self._detect_dive(detections['players'])
            score += dive_score * 3
            if dive_score > 0.5 and score > 5:
                detections['action_type'] = 'dive_save'
            
            # 规则5: 球消失 + 球员停动 → 得分
            if current_ball_center is None and len(self.ball_positions) > 5:
                if self._detect_players_stopped(detections['players']):
                    score += 4
                    detections['action_type'] = 'scoring'
            
            # 综合判定
            detections['has_action'] = score >= 2.0
            detections['highlight_score'] = min(score, 10.0)
        
        # 动作持续性（避免频繁切换）
        self.action_history.append(detections['has_action'])
        if sum(self.action_history) >= 1:
            detections['has_action'] = True
        
        # 更新追踪历史
        if current_ball_center:
            self.ball_positions.append(current_ball_center)
        
        # Debug 可视化
        if debug:
            self._draw_debug(frame, detections)
        
        return detections
    
    def _calc_ball_speed(self, ball_center):
        """计算羽毛球速度（像素/帧）"""
        if ball_center is None or len(self.ball_positions) < 2:
            return 0
        
        prev = self.ball_positions[-1]
        dist = np.sqrt((ball_center[0] - prev[0])**2 + (ball_center[1] - prev[1])**2)
        return dist
    
    def _calc_arm_speed(self, players):
        """估算球员手臂动作速度"""
        if not players:
            return 0
        # 用球员边界框高度变化来近似手臂动作
        heights = [p['bbox_height'] for p in players]
        if len(self.track_history) > 0:
            prev_heights = self.track_history[-1].get('player_heights', heights)
            speed = np.mean([abs(h - ph) for h, ph in zip(heights, prev_heights)])
        else:
            speed = 0
        
        self.track_history.append({'player_heights': heights})
        return speed
    
    def _update_rally_count(self, ball_center):
        """更新多拍计数 - 球在场地两侧交替出现"""
        if ball_center is None:
            return
        
        # 判断球在哪一侧（根据x坐标）
        current_side = 'left' if ball_center[0] < 320 else 'right'
        
        if self.last_ball_side and current_side != self.last_ball_side:
            self.rally_count += 1
        self.last_ball_side = current_side
    
    def _detect_dive(self, players):
        """检测鱼跃救球 - 球员bbox高度快速减小"""
        if not players or len(self.track_history) < 3:
            return 0
        
        # 看球员高度是否快速下降
        current_heights = [p['bbox_height'] for p in players]
        prev_data = self.track_history[-1]
        if 'player_heights' in prev_data:
            prev_heights = prev_data['player_heights']
            if len(current_heights) == len(prev_heights):
                drops = [ph - ch for ph, ch in zip(prev_heights, current_heights)]
                max_drop = max(drops) if drops else 0
                return max(0, max_drop / 50)  # 归一化
        return 0
    
    def _detect_players_stopped(self, players):
        """检测球员是否停止移动（得分信号）"""
        if not players or len(self.track_history) < 5:
            return False
        
        current_centers = {id(p): p['center'] for p in players}
        total_movement = 0
        count = 0
        
        for data in list(self.track_history)[-5:]:
            if 'player_centers' in data:
                for pid, center in data['player_centers'].items():
                    if pid in current_centers:
                        movement = np.sqrt(
                            (current_centers[pid][0] - center[0])**2 +
                            (current_centers[pid][1] - center[1])**2
                        )
                        total_movement += movement
                        count += 1
        
        avg_movement = total_movement / max(count, 1)
        return avg_movement < 5  # 平均移动小于5像素 → 停止
    
    def _draw_debug(self, frame, detections):
        """Debug可视化"""
        debug_frame = frame.copy()
        
        # 画球员
        for p in detections['players']:
            x1, y1, x2, y2 = p['bbox']
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug_frame, f"Player {p['confidence']:.2f}",
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        # 画羽毛球
        if detections['ball']:
            x1, y1, x2, y2 = detections['ball']['bbox']
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(debug_frame, f"Shuttle {detections['ball']['confidence']:.2f}",
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        
        # 画球拍
        for r in detections.get('rackets', []):
            x1, y1, x2, y2 = r['bbox']
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # 状态面板
        status = f"Badminton | Rally: {self.rally_count} | Score: {detections['highlight_score']:.1f}"
        action = detections['action_type'] or 'waiting'
        color = (0, 255, 0) if detections['has_action'] else (0, 0, 255)
        
        cv2.putText(debug_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(debug_frame, f"Action: {action}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if detections['has_action']:
            cv2.putText(debug_frame, "🔴 HIGHLIGHT", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        cv2.imshow("Badminton Detection", debug_frame)
        cv2.waitKey(1)
