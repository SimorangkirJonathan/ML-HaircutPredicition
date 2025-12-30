"""
Enhanced Hair Type Classifier Training with Fine-Tuning
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from pathlib import Path


def create_model(num_classes=5, fine_tune=False):
    """Create MobileNetV2-based model with optional fine-tuning."""
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    if fine_tune:
        # Unfreeze last 30 layers for fine-tuning
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        for layer in base_model.layers[-30:]:
            layer.trainable = True
        print("Fine-tuning enabled: Last 30 layers unfrozen")
    else:
        base_model.trainable = False
    
    # Build model with stronger regularization
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        Dropout(0.5),
        BatchNormalization(),
        Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    return model


def train(data_dir, output_dir, epochs=30, batch_size=16):
    """Train with enhanced data augmentation and fine-tuning."""
    print("=" * 60)
    print("ENHANCED HAIR TYPE CLASSIFIER TRAINING")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Enhanced data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        zoom_range=0.3,
        shear_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.2
    )
    
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    print(f"\nLoading data from: {data_dir}")
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    print(f"\nClasses: {train_generator.class_indices}")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    
    # Phase 1: Train with frozen base model
    print("\n" + "=" * 60)
    print("PHASE 1: Training with frozen base model")
    print("=" * 60)
    
    model = create_model(num_classes=len(train_generator.class_indices), fine_tune=False)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks_phase1 = [
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]
    
    model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        callbacks=callbacks_phase1,
        verbose=1
    )
    
    # Phase 2: Fine-tune
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning with unfrozen layers")
    print("=" * 60)
    
    # Unfreeze last 30 layers
    for layer in model.layers[0].layers[-30:]:
        layer.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks_phase2 = [
        EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True),
        ModelCheckpoint(
            os.path.join(output_dir, 'hair_type_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    ]
    
    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks_phase2,
        verbose=1
    )
    
    # Final save
    model.save(os.path.join(output_dir, 'hair_type_model.keras'))
    
    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    val_loss, val_acc = model.evaluate(val_generator, verbose=0)
    print(f"\nFinal Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"Final Validation Loss: {val_loss:.4f}")
    
    # Save class mapping
    with open(os.path.join(output_dir, 'class_mapping.txt'), 'w') as f:
        for class_name, idx in train_generator.class_indices.items():
            f.write(f"{idx}: {class_name}\n")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Model saved to: {os.path.join(output_dir, 'hair_type_model.keras')}")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    project_dir = Path(__file__).parent
    data_dir = project_dir / "data"
    output_dir = project_dir / "models"
    
    model = train(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        epochs=30,
        batch_size=16
    )
