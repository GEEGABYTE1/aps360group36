import matplotlib.pyplot as plt

# Data from your log
epochs = [1, 2, 3, 4, 5, 6]
train_loss = [1.5283, 0.8431, 0.6834, 0.6023, 0.5210, 0.4443]
val_loss = [1.5972, 1.5356, 1.5302, 1.5158, 1.5043, 1.4577]
train_acc = [0.6149, 0.7199, 0.7452, 0.7665, 0.7955, 0.8256]
val_acc = [0.5951, 0.5749, 0.5995, 0.6006, 0.6255, 0.6437]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, label='Train Acc')
plt.plot(epochs, val_acc, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()