# MRP-FER

Facial expressions of emotion are a major channel in our daily communications, and it has been
subject of intense research in recent years. To automatically infer facial expressions, convolutional
neural network based approaches has become widely adopted due to their proven applicability to
Facial Expression Recognition (FER) task. On the other hand Virtual Reality (VR) has gained popularity
as an immersive multimedia platform, where FER can provide enriched media experiences.
However, recognizing facial expression while wearing a head-mounted VR headset is a challenging
task due to the upper half of the face being completely occluded. In this project we attempt to
overcome these issues and focus on facial expression recognition in presence of a severe occlusion
where the user is wearing a head-mounted display in a VR setting. We propose a geometric model
to simulate occlusion resulting from a Samsung Gear VR headset that can be applied to existing
FER datasets. Then, we adopt a transfer learning approach, starting from two pretrained networks,
namely VGG and ResNet. We further fine-tune the networks on FER+, AffectNet and RAF-DB
datasets. Experimental results show that our approach achieves comparable results to existing
methods while training on three modified benchmark datasets that adhere to realistic occlusion
resulting from wearing a commodity VR headset.
