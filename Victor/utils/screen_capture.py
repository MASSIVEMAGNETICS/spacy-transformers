import numpy as np
import torch
from torchvision import transforms

class ScreenCapture:
    def __init__(self, size=(224, 224)):
        self.size = size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.sct = None

    def _initialize_sct(self):
        """Initializes the mss screen capturer."""
        try:
            import mss
            self.sct = mss.mss()
        except ImportError:
            print("Warning: mss library not found. Screen capture will be disabled. Please run 'pip install mss'.")
            self.sct = None
        except Exception as e:
            print(f"Error initializing screen capture: {e}")
            self.sct = None

    def capture_frame(self) -> torch.Tensor:
        """Capture current screen frame and return as tensor."""
        if self.sct is None:
            self._initialize_sct()
            if self.sct is None:
                # Fallback to a random tensor if initialization fails
                return torch.randn(3, self.size[0], self.size[1])

        try:
            import cv2
            monitor = self.sct.monitors[1]  # Primary monitor
            screenshot = self.sct.grab(monitor)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            return self.transform(img)
        except ImportError:
            print("Warning: opencv-python not found. Screen capture will be disabled. Please run 'pip install opencv-python'.")
            return torch.randn(3, self.size[0], self.size[1])
        except Exception as e:
            print(f"Error during screen capture: {e}")
            return torch.randn(3, self.size[0], self.size[1])
