```python
import tarfile
from pathlib import Path

import numpy as np
import tensorflow as tf

# TensorFlow 官方花卉数据集。第一次运行时会自动下载，之后会复用本地缓存。
FLOWER_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

print("TensorFlow 版本:", tf.__version__)
```

    TensorFlow 版本: 2.21.0
    


```python
# 数据目录配置：
# - DATA_DIR = None：自动下载并使用 TensorFlow 官方 flowers 数据集。
# - DATA_DIR = r"D:\path\to\my_images"：使用你自己的图片分类目录。
#
# 自定义图片目录需要按类别分文件夹，例如：
# my_images/
#   daisy/
#     1.jpg
#   roses/
#     2.jpg
DATA_DIR = None

# 导出目录。训练完成后会在这里生成 model.tflite、labels.txt 和 flower_classifier.keras。
EXPORT_DIR = "exported_flower_model"

# 训练参数。教程演示可以先用 3 到 5 个 epoch；如果使用自己的数据，可以适当增加。
EPOCHS = 5
BATCH_SIZE = 32
IMAGE_SIZE = 224
LEARNING_RATE = 1e-3

# TFLite 量化方式：
# - "dynamic"：默认推荐，模型更小，通常最容易成功。
# - "float16"：适合部分支持 float16 的设备。
# - "int8"：体积更小，但需要代表性数据集，转换要求更严格。
# - "none"：不量化，保留浮点模型。
QUANTIZATION = "dynamic"

# 固定随机种子，方便训练/验证划分尽量可复现。
SEED = 123
```


```python
def load_flower_datasets(data_dir, image_size, batch_size, seed):
    # 如果没有传入自定义数据目录，就下载 TensorFlow 官方 flower_photos 数据集。
    if data_dir is None:
        archive_path = tf.keras.utils.get_file(
            "flower_photos.tgz",
            FLOWER_URL,
            extract=False,
        )
        archive_path = Path(archive_path)

        # Keras 可能已经缓存了解压后的目录；先检查常见位置，避免重复解压。
        candidates = [
            archive_path.parent / "flower_photos",
            archive_path.parent / "flower_photos_extracted" / "flower_photos",
        ]
        data_dir = next((path for path in candidates if path.exists()), None)
        if data_dir is None:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(archive_path.parent / "flower_photos_extracted")
            data_dir = archive_path.parent / "flower_photos_extracted" / "flower_photos"
    else:
        data_dir = Path(data_dir)

    # 从目录读取图片。目录下的每个子文件夹会被当作一个类别。
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size,
    )
    class_names = train_ds.class_names

    # 原始 validation 部分再拆成验证集和测试集：验证集用于训练过程中观察效果，测试集用于最后评估。
    val_batches = int(tf.data.experimental.cardinality(val_ds).numpy())
    test_ds = val_ds.take(val_batches // 2)
    val_ds = val_ds.skip(val_batches // 2)

    # cache/prefetch 可以减少数据读取等待；shuffle 只用于训练集。
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=seed).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names
```


```python
# 加载数据集并查看类别名称。
train_ds, val_ds, test_ds, class_names = load_flower_datasets(
    DATA_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED,
)

print("类别数量:", len(class_names))
print("类别名称:", class_names)
```

    Found 3670 files belonging to 5 classes.
    Using 2936 files for training.
    Found 3670 files belonging to 5 classes.
    Using 734 files for validation.
    类别数量: 5
    类别名称: ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    


```python
def build_model(num_classes, image_size, learning_rate):
    # 输入图片尺寸固定为 IMAGE_SIZE x IMAGE_SIZE x 3。
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")

    # MobileNetV2 有自己的预处理方式，这里把像素值转换到模型期望的范围。
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # include_top=False 表示不要 ImageNet 原始的 1000 类分类头，只保留特征提取部分。
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )

    # 冻结预训练模型参数，只训练后面的 Dense 分类层。
    base_model.trainable = False
    x = base_model(x, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)

    # 输出维度等于类别数量，softmax 输出每个类别的概率。
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model
```


```python
# 创建模型并打印结构。第一次运行会下载 MobileNetV2 的 ImageNet 预训练权重。
model = build_model(len(class_names), IMAGE_SIZE, LEARNING_RATE)
model.summary()
```

    WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
    


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ image (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ true_divide (<span style="color: #0087ff; text-decoration-color: #0087ff">TrueDivide</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ subtract (<span style="color: #0087ff; text-decoration-color: #0087ff">Subtract</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ mobilenetv2_1.00_224            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1280</span>)           │     <span style="color: #00af00; text-decoration-color: #00af00">2,257,984</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Functional</span>)                    │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1280</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ predictions (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">5</span>)              │         <span style="color: #00af00; text-decoration-color: #00af00">6,405</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">2,264,389</span> (8.64 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">6,405</span> (25.02 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">2,257,984</span> (8.61 MB)
</pre>




```python
# 开始训练。history 中会保存每个 epoch 的 loss、accuracy、val_loss、val_accuracy。
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
```

    Epoch 1/5
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m32s[0m 296ms/step - accuracy: 0.6635 - loss: 0.8789 - val_accuracy: 0.8534 - val_loss: 0.4375
    Epoch 2/5
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 272ms/step - accuracy: 0.8525 - loss: 0.4221 - val_accuracy: 0.8796 - val_loss: 0.3528
    Epoch 3/5
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m26s[0m 279ms/step - accuracy: 0.8839 - loss: 0.3441 - val_accuracy: 0.8927 - val_loss: 0.3053
    Epoch 4/5
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 272ms/step - accuracy: 0.8995 - loss: 0.2893 - val_accuracy: 0.9031 - val_loss: 0.2828
    Epoch 5/5
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 277ms/step - accuracy: 0.9189 - loss: 0.2562 - val_accuracy: 0.9084 - val_loss: 0.2666
    


```python
# 使用测试集评估模型。测试集没有参与训练，用于更客观地观察最终效果。
loss, accuracy = model.evaluate(test_ds)
print(f"test_loss={loss:.4f}, test_accuracy={accuracy:.4f}")
```

    [1m11/11[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 246ms/step - accuracy: 0.8949 - loss: 0.3197
    test_loss=0.3197, test_accuracy=0.8949
    


```python
def convert_to_tflite(model, quantization, representative_ds):
    # 从 Keras 模型创建 TFLite 转换器。
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantization == "dynamic":
        # 动态范围量化：最常用、最容易成功的压缩方式。
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization == "float16":
        # float16 量化：权重使用半精度浮点数，适合部分移动端/GPU 场景。
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == "int8":
        # int8 全整数量化：体积更小，但需要代表性数据集校准输入分布。
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        def representative_data_gen():
            for images, _ in representative_ds.take(100):
                for image in images:
                    yield [tf.expand_dims(tf.cast(image, tf.float32), 0)]

        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    elif quantization != "none":
        raise ValueError(f"Unsupported quantization mode: {quantization}")

    return converter.convert()
```


```python
# 创建导出目录。
export_dir = Path(EXPORT_DIR)
export_dir.mkdir(parents=True, exist_ok=True)

# 保存标签文件。部署时需要 labels.txt 把模型输出编号映射回类别名称。
labels_path = export_dir / "labels.txt"
labels_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")

# 保存 Keras 原始模型，便于以后继续训练或重新转换。
keras_path = export_dir / "flower_classifier.keras"
model.save(keras_path)

# 转换并保存 TFLite 模型。
tflite_model = convert_to_tflite(model, QUANTIZATION, train_ds)
tflite_path = export_dir / "model.tflite"
tflite_path.write_bytes(tflite_model)

print(f"已保存 Keras 模型: {keras_path}")
print(f"已保存 TFLite 模型: {tflite_path}")
print(f"已保存标签文件: {labels_path}")
```

    INFO:tensorflow:Assets written to: C:\Users\30280\AppData\Local\Temp\tmpyhho0s14\assets
    

    INFO:tensorflow:Assets written to: C:\Users\30280\AppData\Local\Temp\tmpyhho0s14\assets
    

    Saved artifact at 'C:\Users\30280\AppData\Local\Temp\tmpyhho0s14'. The following endpoints are available:
    
    * Endpoint 'serve'
      args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name='image')
    Output Type:
      TensorSpec(shape=(None, 5), dtype=tf.float32, name=None)
    Captures:
      1425793528464: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425793529040: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425793528848: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425793529424: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425793528272: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425793529616: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425793528656: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425793530192: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425793526352: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425793527888: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425793529808: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425793530960: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425793530768: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425793531152: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425793530384: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425793531344: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425793531728: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425793531536: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425793530576: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425793529232: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794778320: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794779472: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794779664: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794778896: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794778512: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794780624: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794781200: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794781008: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794780240: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794781392: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794781776: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794782544: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794782352: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794782736: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794780432: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794781968: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794783312: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794783120: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794782160: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794783504: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794783888: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794784464: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794784272: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794782928: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794784656: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794785040: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794785808: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794785616: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794786000: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794780816: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794785232: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794786768: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794786576: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794786960: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794784080: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794786192: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794787728: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794787536: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794787920: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794785424: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794787152: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794788688: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794788496: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794788880: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794786384: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794788112: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794789648: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794789456: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794789840: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794787344: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794789072: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794790608: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794790416: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794790800: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794788304: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794790992: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794791760: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794791568: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794791952: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794790032: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794792144: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794792912: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794792720: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794793104: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794791184: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794792336: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794789264: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799266576: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794792528: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425794793296: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799266960: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799267728: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799267536: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799267920: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799266384: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799267152: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799268688: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799268496: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799268880: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799266768: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799268112: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799269648: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799269456: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799269840: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799267344: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799269072: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799270608: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799270416: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799270800: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799268304: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799270032: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799271568: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799271376: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799271760: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799269264: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799270992: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799272528: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799272336: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799272720: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799270224: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799271952: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799273488: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799273296: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799273680: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799271184: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799272912: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799274448: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799274256: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799274640: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799272144: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799273872: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799275408: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799275216: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799275600: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799273104: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799274832: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799276368: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799276176: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799276560: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799274064: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799275792: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799277328: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799277136: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799277520: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799275024: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799276752: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799278288: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799278096: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799278480: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799275984: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799277712: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799279248: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799279056: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799279440: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799276944: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799278672: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799280208: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799280016: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799280400: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799277904: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799279632: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799281168: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799280976: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799281360: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799278864: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799280592: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799282128: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799281936: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799282320: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799279824: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799281552: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799280784: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800118544: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799281744: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425799282512: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800118928: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800119696: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800119504: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800119888: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800118352: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800119120: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800120656: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800120464: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800120848: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800118736: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800120080: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800121616: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800121424: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800121808: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800119312: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800121040: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800122576: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800122384: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800122768: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800120272: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800122000: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800123536: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800123344: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800123728: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800121232: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800122960: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800124496: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800124304: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800124688: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800122192: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800123920: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800125456: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800125264: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800125648: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800123152: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800124880: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800126416: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800126224: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800126608: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800124112: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800125840: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800127376: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800127184: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800127568: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800125072: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800126800: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800128336: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800128144: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800128528: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800126032: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800127760: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800129296: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800129104: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800129488: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800126992: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800128720: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800130256: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800130064: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800130448: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800127952: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800129680: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800131216: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800131024: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800131408: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800128912: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800130640: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800132176: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800131984: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800132368: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800129872: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800131600: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800133136: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800132944: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800133328: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800130832: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800132560: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800134096: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800133904: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800134288: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800131792: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800133520: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800132752: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425824055568: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800133712: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425800134480: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425824056528: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1425824057488: TensorSpec(shape=(), dtype=tf.resource, name=None)
    已保存 Keras 模型: exported_flower_model\flower_classifier.keras
    已保存 TFLite 模型: exported_flower_model\model.tflite
    已保存标签文件: exported_flower_model\labels.txt
    


```python
def smoke_test_tflite(tflite_path, test_ds, class_names):
    # 加载 TFLite 模型并分配张量内存。
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # 从测试集中取 8 张图片做快速推理。
    images, labels = next(iter(test_ds.unbatch().batch(8)))
    input_data = tf.cast(images, input_details["dtype"]).numpy()

    # 如果模型是 uint8 输入，需要按照量化参数把图片转换到对应范围。
    if input_details["dtype"] == np.uint8:
        scale, zero_point = input_details["quantization"]
        if scale:
            input_data = images.numpy() / scale + zero_point
            input_data = np.clip(input_data, 0, 255).astype(np.uint8)

    predictions = []
    for image in input_data:
        interpreter.set_tensor(input_details["index"], np.expand_dims(image, 0))
        interpreter.invoke()
        predictions.append(interpreter.get_tensor(output_details["index"])[0])

    predicted_ids = np.argmax(np.asarray(predictions), axis=1)
    for expected, predicted in zip(labels.numpy()[:5], predicted_ids[:5]):
        print(f"真实类别={class_names[expected]}, 预测类别={class_names[predicted]}")
```


```python
# 运行 TFLite 快速测试。
smoke_test_tflite(tflite_path, test_ds, class_names)
```

    E:\Anaconda_envs\envs\TensorFlow\Lib\site-packages\tensorflow\lite\python\interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
        TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
        See the [migration guide](https://ai.google.dev/edge/litert/migration)
        for details.
        
      warnings.warn(_INTERPRETER_DELETION_WARNING)
    

    真实类别=daisy, 预测类别=daisy
    真实类别=tulips, 预测类别=tulips
    真实类别=sunflowers, 预测类别=sunflowers
    真实类别=daisy, 预测类别=daisy
    真实类别=sunflowers, 预测类别=sunflowers
    


```python

```
