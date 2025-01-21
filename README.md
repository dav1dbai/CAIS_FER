## CAIS Winter Project: Facial Emotion Recognition
David Bai, dmbai@usc.edu

I built a convolutional neural network to classify emotions on various photos of faces, achieving 61% accuracy (with random choice being 16%) on sorting images into 5 different categories.

The dataset is Kaggle's Facial Emotion Recognition Dataset which contains a 28.709 image train set and 3,589 images on the train set. Each image is a 48x48 black and white image with the face approximately centered.

# Preprocessing and hyperparameters

- Preprocessing: I tried various transformations on the faces in the training set (rotations, flips, etc) to improve diversity but the final version ultimately did not include it due to its minimal impact on performance. This is likely because the majority of the diversity in the data is a result of the face— the approximat location and orientation of the face, which is what we have the power to affect, is usually pretty consistent across data points.

- Hyperparameters: I used Cross-Entropy Loss and AdamW with a LR of 0.0001 for my loss functions and optimizer, respectively. Again, these were adjusted when playing with the training set and these seemed to work best. I split the large train set into a train and validation of 70/30 to avoid tuning off the test set. I expect changing this ratio in favor of the train subset would've improved performance minimally, because I stratified the split for even class distribution.
  
# Model architecture

This had the greatest impact on model performance and I went through several iterations here. At first, I tested a couple different versions of a CNN with increasing or decreasing numbers of convolutional layers, average vs max pooling. Models that downsampled the already pretty low-res images would get stuck at around ~30-40% accuracy. The polar opposite of that— just feeding the flattened image into a MLP, also didn't yield good results and also quickly faced a ceiling during training. 

Ultimately, what worked for me was adding skip connections ResNet-style which helped boost the accuracy a little bit, and is the current implementation. 

# Results

Here's the accuracy and F1 score of the test set by class:
Test Loss: 1.0681, Test Acc: 0.6116, Weighted F1: 0.6104
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|---------|----------|----------|
| angry | 0.51 | 0.53 | 0.52 | 958 |
| disgust | 0.68 | 0.49 | 0.57 | 111 |
| fear | 0.46 | 0.36 | 0.40 | 1024 |
| happy | 0.84 | 0.82 | 0.83 | 1774 |
| sad | 0.45 | 0.50 | 0.47 | 1247 |
| surprise | 0.75 | 0.75 | 0.75 | 831 |
| neutral | 0.56 | 0.62 | 0.59 | 1233 |
| accuracy | - | - | 0.61 | 7178 |
| macro avg | 0.61 | 0.58 | 0.59 | 7178 |
| weighted avg | 0.61 | 0.61 | 0.61 | 7178 |

We can roughly see that F1 score correlates intuitively with how expressive certain emotions are (happy, surprise). This makes sense— these are the easiest features most resistant to downsampling. A wide smile wil occupy and light up more neurons when it comes time for the forward pass. This is also supported by the high F1 score of disgust relative to its low representation in the dataset.


# Discussion

- Performance-wise, I'm not very happy with the model. I think I could spend more time playing around with the architecture as well as trying to much better understand the data, but I hit time constraints. Next steps here would include transfer-learning something like ResNet to see what the baseline might be, and exploring how we can extract more information from the images.  

- Application-wise, I think FER has a lot of potential for social good. Consider applications such as those being researched in the Interaction Lab @USC, where the ability to recognize emotions for educational or therapeutic purposes could prove extremely useful. Obviously, though, this is one of those situations where  a false positive or negative could have real consequences so that's something to consider. Next steps here would be determining how fast we can run inference on the model for real-time situations. 
