"""
Core Deepfake Detection Engine
Handles model loading, inference, and post-processing
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import logging
from collections import deque
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DeepfakeDetectionModel(nn.Module):
    """
    EfficientNet-B5 based detector with custom classification head
    """
    def __init__(self, pretrained=True, num_classes=1):
        super(DeepfakeDetectionModel, self).__init__()
        
        # Feature extractor backbone
        from torchvision.models import efficientnet_b5
        efficientnet = efficientnet_b5(pretrained=pretrained)
        self.features = efficientnet.features
        
        # Get feature dimension
        self.feature_dim = efficientnet.classifier[1].in_features
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
        
    def forward(self, x):
        # Feature extraction
        features = self.features(x)
        features = features.mean([2, 3])  # Global average pooling
        
        # Classification
        logits = self.classifier(features)
        return logits


class FaceDetector:
    """
    MTCNN-based face detection with fallback to MediaPipe
    """
    def __init__(self, device='cuda'):
        self.device = device
        try:
            from facenet_pytorch import MTCNN
            self.mtcnn = MTCNN(keep_all=True, device=device)
            self.use_mtcnn = True
        except ImportError:
            logger.warning("MTCNN not available, falling back to MediaPipe")
            import mediapipe as mp
            self.mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
            self.use_mtcnn = False
    
    def detect(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """
        Detect faces in frame
        
        Args:
            frame: Input frame (H, W, 3) in BGR
            
        Returns:
            boxes: List of face bounding boxes
            confidences: Detection confidence scores
        """
        if self.use_mtcnn:
            return self._detect_mtcnn(frame)
        else:
            return self._detect_mediapipe(frame)
    
    def _detect_mtcnn(self, frame: np.ndarray) -> Tuple[List, List]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs = self.mtcnn.detect(rgb_frame, landmarks=False)
        
        if boxes is None:
            return [], []
        
        return boxes.tolist(), probs.tolist()
    
    def _detect_mediapipe(self, frame: np.ndarray) -> Tuple[List, List]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face.process(rgb_frame)
        
        boxes, confidences = [], []
        if results.detections:
            h, w, _ = frame.shape
            for detection in results.detections:
                bbox = detection.location_data.bounding_box
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                boxes.append([x, y, x + width, y + height])
                confidences.append(detection.score[0])
        
        return boxes, confidences


class PreprocessingPipeline:
    """
    Face preprocessing: alignment, normalization, augmentation
    """
    def __init__(self, img_size=256, device='cuda'):
        self.img_size = img_size
        self.device = device
        
        # ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    def preprocess_face(self, face_img: np.ndarray, augment=False) -> torch.Tensor:
        """
        Preprocess face image
        
        Args:
            face_img: Face image (H, W, 3) in BGR
            augment: Apply augmentation
            
        Returns:
            Preprocessed tensor (1, 3, 256, 256)
        """
        # Resize
        face = cv2.resize(face_img, (self.img_size, self.img_size))
        
        # Convert BGR to RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Optional augmentation
        if augment:
            face = self._augment(face)
        
        # Convert to tensor
        face = torch.from_numpy(face).float() / 255.0
        face = face.permute(2, 0, 1).unsqueeze(0)
        
        # Normalize
        face = (face - self.mean.to(self.device)) / self.std.to(self.device)
        
        return face.to(self.device)
    
    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Light augmentation for inference"""
        # Small rotation
        angle = np.random.uniform(-10, 10)
        h, w = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img = cv2.warpAffine(img, matrix, (w, h))
        
        return img
    
    def extract_face(self, frame: np.ndarray, bbox: List) -> np.ndarray:
        """
        Extract and align face from frame
        
        Args:
            frame: Full frame
            bbox: [x1, y1, x2, y2]
            
        Returns:
            Cropped face image
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add padding
        h, w = frame.shape[:2]
        pad = int(0.1 * (x2 - x1))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        face = frame[y1:y2, x1:x2]
        return face


class DeepfakeDetector:
    """
    Main deepfake detection system
    """
    def __init__(
        self,
        model_path: str = 'models/detector.pth',
        device: str = 'cuda',
        confidence_threshold: float = 0.5,
        batch_size: int = 8
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        
        # Initialize components
        self.model = DeepfakeDetectionModel()
        self._load_model(model_path)
        self.model.eval()
        self.model.to(device)
        
        self.face_detector = FaceDetector(device=device)
        self.preprocessor = PreprocessingPipeline(device=device)
        
        # Temporal smoothing buffer
        self.prediction_buffer = deque(maxlen=5)
        
        logger.info("DeepfakeDetector initialized successfully")
    
    def _load_model(self, model_path: str):
        """Load pretrained weights"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state'])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info(f"Model loaded from {model_path}")
        except FileNotFoundError:
            logger.warning(f"Model not found at {model_path}. Using untrained model.")
    
    def detect_video(
        self,
        video_path: str,
        output_dir: str = './results',
        output_video: bool = True,
        skip_frames: int = 2
    ) -> Dict:
        """
        Detect deepfakes in video
        
        Args:
            video_path: Path to video file
            output_dir: Output directory for results
            output_video: Generate annotated video
            skip_frames: Process every nth frame
            
        Returns:
            results: Detection results
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Output video writer
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                f"{output_dir}/detected_{Path(video_path).name}",
                fourcc, fps, (width, height)
            )
        
        frame_predictions = []
        frame_count = 0
        
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % skip_frames != 0:
                    frame_count += 1
                    continue
                
                # Detect faces
                boxes, confidences = self.face_detector.detect(frame)
                
                if not boxes:
                    frame_count += 1
                    continue
                
                # Process each face
                face_predictions = []
                for box, conf in zip(boxes, confidences):
                    if conf < 0.5:  # Low face detection confidence
                        continue
                    
                    face = self.preprocessor.extract_face(frame, box)
                    pred = self._predict_face(face)
                    face_predictions.append(pred)
                    
                    # Draw on frame
                    if output_video:
                        frame = self._draw_detection(
                            frame, box, pred, conf
                        )
                
                if face_predictions:
                    avg_pred = np.mean(face_predictions)
                    frame_predictions.append(avg_pred)
                
                if output_video:
                    out.write(frame)
                
                frame_count += 1
                
                if frame_count % 30 == 0:
                    logger.info(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        if output_video:
            out.release()
        
        # Aggregate results
        if not frame_predictions:
            return {
                'verdict': 'INCONCLUSIVE',
                'confidence': 0.0,
                'frames_processed': frame_count,
                'faces_detected': 0,
                'heatmap_path': None
            }
        
        avg_confidence = np.mean(frame_predictions)
        verdict = 'FAKE' if avg_confidence > self.confidence_threshold else 'REAL'
        
        return {
            'verdict': verdict,
            'confidence': float(avg_confidence),
            'frames_processed': frame_count,
            'faces_detected': len(frame_predictions),
            'frame_predictions': frame_predictions,
            'output_video': f"{output_dir}/detected_{Path(video_path).name}",
            'status': 'SUCCESS'
        }
    
    def _predict_face(self, face: np.ndarray) -> float:
        """
        Get prediction for single face
        
        Args:
            face: Face image (H, W, 3)
            
        Returns:
            Probability of being fake (0-1)
        """
        with torch.no_grad():
            preprocessed = self.preprocessor.preprocess_face(face, augment=False)
            logits = self.model(preprocessed)
            prob = torch.sigmoid(logits).item()
        
        return prob
    
    def _draw_detection(
        self,
        frame: np.ndarray,
        bbox: List,
        prediction: float,
        face_conf: float
    ) -> np.ndarray:
        """Draw detection results on frame"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Color based on prediction
        if prediction > self.confidence_threshold:
            color = (0, 0, 255)  # Red for fake
            label = f"FAKE ({prediction:.2f})"
        else:
            color = (0, 255, 0)  # Green for real
            label = f"REAL ({1-prediction:.2f})"
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Draw label
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            frame, label, (x1, y1 - 10),
            font, 1.0, color, 2
        )
        
        return frame


# Configuration
class Config:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_PATH = 'models/detector.pth'
    CONFIDENCE_THRESHOLD = 0.5
    BATCH_SIZE = 8
    IMG_SIZE = 256
    FPS_TARGET = 30
    
    # Data settings
    TRAIN_DATA_DIR = 'data/train'
    VAL_DATA_DIR = 'data/val'
    TEST_DATA_DIR = 'data/test'


if __name__ == '__main__':
    # Example usage
    detector = DeepfakeDetector(
        model_path=Config.MODEL_PATH,
        device=Config.DEVICE,
        confidence_threshold=Config.CONFIDENCE_THRESHOLD
    )
    
    results = detector.detect_video(
        video_path='sample_video.mp4',
        output_dir='./results'
    )
    
    print(f"Verdict: {results['verdict']}")
    print(f"Confidence: {results['confidence']:.2%}")
