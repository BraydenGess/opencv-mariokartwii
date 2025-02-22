# Development 

The development directory contains the core machine learning models and training scripts for the MarioKart Wii project:

### Flag Detection Model
Located in `flag_detection.py`, this model focuses on binary classification to detect whether a checkpoint flag is present in a frame. Key features:

- Uses a lightweight ResNet-18 architecture modified for binary classification (flag/no flag)
- Input images are resized to 128x128 pixels
- Includes both training and inference code
- Saves models with timestamps in the models/ directory
- Includes a custom FlagDataset class for loading flag/no-flag images

### Course Detection Model
Located in `train.py`, this model performs multi-class classification to identify different MarioKart courses. Key features:

- Handles 34 different classes:
  - None (no course)
  - Opening
  - All MarioKart wii courses
    - Mushroom Cup (Luigi Circuit, Moo Meadows, Mushroom Gorge, Toad's Factory)
    - Flower Cup (Mario Circuit, Coconut Mall, DK Summit, Wario’s Gold Mine)
    - Star Cup (Daisy Circuit, Koopa Cape, Maple Treeway, Grumble Volcano)
    - Special Cup (Dry Dry Ruins, Moonview Highway, Bowser’s Castle, Rainbow Road)
    - Shell Cup (GCN Peach Beach, DS Yoshi Falls, SNES Ghost Valley 2, N64 Mario Raceway_
    - Banana Cup (N64 Sherbet Land, GBA Shy Guy Beach, DS Delfino Square, GCN Waluigi Stadium)
    - Leaf Cup (DS Desert Hills, GBA Bowser Castle 3, N64 DK’s Jungle Parkway, GCN Mario Circuit)
    - Lightning Cup (SNES Mario Circuit 3, DS Peach Gardens, GCN DK Mountain, N64 Bowser’s Castle)
- Uses a custom MarioKartDataset class for loading course images
- Training includes validation to prevent overfitting

### Evaluation Tools

The `flag_metrics.py` script provides comprehensive evaluation of the flag detection model:

- Generates confusion matrices
- Calculates precision, recall, and F1 scores
- Measures inference speed and FPS
- Creates detailed evaluation reports
- Generates ROC curves and AUC scores

The models work together to provide both course identification and checkpoint detection during gameplay. The flag detection runs at high FPS for real-time feedback, while the course detection helps track progress through different tracks.




