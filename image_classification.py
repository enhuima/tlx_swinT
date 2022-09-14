from tlxzoo.module import *
import tensorlayerx as tlx


class ImageClassification(tlx.nn.Module):
    def __init__(self, backbone, l2_weights=False, **kwargs):
        super(ImageClassification, self).__init__()
        if backbone in ["vgg11", "vgg13", "vgg16", "vgg19"]:
            layer_type = kwargs.pop("layer_type", backbone)
            batch_norm = kwargs.pop("batch_norm", True)
            end_with = kwargs.pop("end_with", "fc1_relu")
            num_labels = kwargs.pop("num_labels", 1000)
            self.backbone = VGG(layer_type, batch_norm, end_with, num_labels=num_labels)
        elif backbone == "resnet50":
            input_shape = kwargs.pop("input_shape", None)
            num_labels = kwargs.pop("num_labels", 1000)
            self.backbone = ResNet50(input_shape=input_shape, num_labels=num_labels, include_top=True)
        elif backbone == "swinT":
            # input_shape = kwargs.pop("input_shape", None)
            num_labels = kwargs.pop("num_labels", 10)
            self.backbone = SwinTransformer(img_size=32,num_classes=num_labels,window_size=4)
        else:
            raise ValueError(f"tlxzoo don`t support {backbone}")

        # self.train_weights = self.trainable_weights
      
    def loss_fn(self, output, target, name="", **kwargs):
        loss = tlx.losses.softmax_cross_entropy_with_logits(output, target)

        # for w in self.l2_weights:
        #     loss += self.li_regularizer(w)
        return loss

    def forward(self, inputs):
        return self.backbone(inputs)

    def predict(self, inputs):
        self.set_eval()
        out = self.backbone(inputs)
        return tlx.argmax(out, axis=-1)






