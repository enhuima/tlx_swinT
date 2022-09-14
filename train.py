import sys
import os
os.environ['TL_BACKEND'] = 'torch'
tlxzoo_path = os.path.abspath(os.path.join('TLX/TLXZoo-main/'))
sys.path.append(tlxzoo_path)
print(sys.path)

from tlxzoo.datasets import DataLoaders
from tlxzoo.vision.transforms import BaseVisionTransform
from image_classification import ImageClassification
import tensorlayerx as tlx

tlx.set_device(device='GPU',id=1)

if __name__ == '__main__':
    cifar10 = DataLoaders("Cifar10", per_device_train_batch_size=128, per_device_eval_batch_size=128)
    # transform = BaseVisionTransform(do_resize=False, do_normalize=True, mean=(125.31, 122.95, 113.86),
    #                                 std=(62.99, 62.09, 66.70))
    transform = BaseVisionTransform(do_resize=False, do_normalize=True, mean=(0.5,0.5,0.5),
                                    std=(0.5,0.5,0.5))

    cifar10.register_transform_hook(transform)
    # print(cifar10.dataset_dict["train"].transforms[-1].is_train)
    # print(cifar10.dataset_dict["test"].transforms[-1].is_train)

    model = ImageClassification(backbone="swinT", l2_weights=True, num_labels=10)

    #weight_decay: l2 regularizition
    optimizer = tlx.optimizers.Adam(lr=0.01, weight_decay=5e-4)
    metric = tlx.metrics.Accuracy()

    n_epoch = 500

    trainer = tlx.model.Model(network=model, loss_fn=model.loss_fn, optimizer=optimizer, metric=metric)
    trainer.train(n_epoch=n_epoch, train_dataset=cifar10.train, test_dataset=cifar10.test, print_freq=1)
    # trainer.eval(test_dataset=cifar10.test)

    model.save_weights("./demo/vision/image_classification/swinT/model.npz")

