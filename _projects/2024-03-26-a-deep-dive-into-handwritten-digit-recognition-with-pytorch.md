---
title: Deciphering Digits- A Deep Dive into Handwritten Digit Recognition with PyTorch
date: 2024-03-15 08:01:35 +0300
label: Machine Learning
image: "/images/project-12.jpg"
featured: true
---

In a world awash with digital data, the ability to automatically recognize handwritten digits stands as a cornerstone for countless applications, from bank check processing to form data entry. My project embarks on a journey through the realms of machine learning to construct a multi-layer perceptron model capable of deciphering digits from the renowned MNIST dataset. This venture not only illuminates the intricacies of neural networks but also showcases the power of PyTorch in unlocking these mysteries.

### Technologies Used

- **Programming Language**: Python
- **Libraries and Frameworks**: PyTorch for modeling, Numpy for numerical computations, Matplotlib and Seaborn for data visualization
- **Dataset**: MNIST Handwritten Digit Dataset

<div class="gallery-box">
  <div class="gallery">
    <img src="/images/2024-03-26-a-deep-dive-into-handwritten-digit-recognition-with-pytorch/project-mnist-data.png" loading="lazy" alt="Project">
  </div>
  <em>MNIST Handwritten Dataset</em>
</div>

### Approach and Implementation

**Data Preprocessing**: The project commenced with loading the MNIST dataset, followed by meticulous preprocessing to split it into distinct sets for training, validation, and testing. This foundational step ensured a robust framework for evaluating model performance.

**Model Architecture**: At the heart of this endeavor lies a customizable multi-layer perceptron architecture. By engineering a flexible model, I was able to experiment with varying depths (num_layers), neuron counts (layer_size), and a palette of activation functions, including the likes of ReLU, Sigmoid, and Tanh. This versatility paved the way for rigorous hyperparameter tuning.

```python
class NeuralNetwork(nn.Module):
    def __init__(self, num_layers, layer_size, activation):
        super(NeuralNetwork, self).__init__()
        layers = [nn.Flatten()]
        for i in range(num_layers - 1):
            layers.append(nn.Linear(layer_size if i > 0 else 28*28, layer_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(layer_size, 10))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.layers(x)
        return logits
```

**Hyperparameter Tuning**: The quest for the optimal model configuration led me through a labyrinth of possibilities, adjusting parameters such as the number of layers, layer sizes, activation functions, weight decay, and training durations (epochs). This exhaustive search was instrumental in honing the model to its finest.

**Training Process**: The training phase was both a challenge and a revelation, with each epoch shedding light on the model's evolving comprehension of handwritten digits. Despite obstacles, strategic adjustments and perseverance bore fruit, culminating in a model of commendable accuracy.

**Visualization**: The journey was visually documented through a series of compelling graphs and images, from the evolution of loss and accuracy over epochs to the intricate details of the confusion matrix and poignant examples of misclassified digits. These visual narratives provided invaluable insights into the model's learning process and areas for refinement.

<!-- ![iPad](images/2024-03-26-a-deep-dive-into-handwritten-digit-recognition-with-pytorch/project-visualization-confusion-matrix.png)

_Photo by [Balázs Kétyi](https://unsplash.com/@balazsketyi) on [Unsplash](https://unsplash.com/)_ -->

<div class="gallery-box">
  <div class="gallery">
    <img src="/images/2024-03-26-a-deep-dive-into-handwritten-digit-recognition-with-pytorch/project-visualization-confusion-matrix.png" loading="lazy" alt="Project">
  </div>
  <em>Confusion Matrix</em>
</div>

### Results

The culmination of this project was marked by a model that achieved a remarkable accuracy of **96.9%** on the test set, a testament to the efficacy of the chosen hyperparameters:

- **Number of Layers**: 4
- **Layer Size**: 512
- **Activation Function**: Leaky ReLU
- **Weight Decay**: 0.01
- **Epochs**: 20

This achievement not only signifies the model's proficiency in digit recognition but also highlights the potential for further exploration and optimization.

> Design is an opportunity to continue telling the story, not just to sum everything up. - Tate Linden

### Lessons Learned and Next Steps

This project was a profound learning experience, providing deep insights into the nuances of neural network optimization and the critical role of hyperparameter tuning. Looking ahead, I am intrigued by the prospect of experimenting with more complex architectures, such as convolutional neural networks, and tackling larger, more diverse datasets.

### Code and Resources

Embark on your own exploration of digit recognition by diving into the [project repository](#).
