"""
Advanced Hair Type Classifier Training
Prevents overfitting with proper regularization, dropout, and data augmentation.
Uses learning rate scheduling and early stopping for optimal training.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, 
    BatchNormalization, GaussianNoise
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, 
    ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.regularizers import l2
from pathlib import Path
import numpy as np


def create_advanced_model(num_classes=5, dropout_rate=0.5, l2_reg=0.01):
    """Create model with strong regularization to prevent overfitting."""
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze all layers initially
    base_model.trainable = False
    
    # Build model with regularization
    model = Sequential([
        # Input noise for regularization
        GaussianNoise(0.1),
        
        # Pre-trained base
        base_model,
        
        # Global pooling
        GlobalAveragePooling2D(),
        
        # Dense layers with regularization
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(256, activation='relu', kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate * 0.8),
        
        Dense(128, activation='relu', kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate * 0.6),
        
        Dense(64, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate * 0.4),
        
        # Output
        Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model


def get_data_generators(data_dir, batch_size=16, validation_split=0.2):
    """Create data generators with strong augmentation."""
    
    # Training augmentation - strong to prevent overfitting
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.25,
        height_shift_range=0.25,
        horizontal_flip=True,
        zoom_range=0.25,
        shear_range=0.15,
        brightness_range=[0.7, 1.3],
        channel_shift_range=20,
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    # Validation - no augmentation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_gen, val_gen


def train_advanced(data_dir, output_dir, epochs=50, batch_size=16):
    """Train with two-phase approach: frozen then fine-tuning."""
    
    print("=" * 60)
    print("ADVANCED HAIR TYPE CLASSIFIER TRAINING")
    print("With Anti-Overfitting Techniques")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data
    train_gen, val_gen = get_data_generators(data_dir, batch_size)
    
    print(f"\nClasses: {train_gen.class_indices}")
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    
    num_classes = len(train_gen.class_indices)
    
    # ==========================================
    # PHASE 1: Train classifier head only
    # ==========================================
    print("\n" + "=" * 60)
    print("PHASE 1: Training classifier head (base frozen)")
    print("=" * 60)
    
    model, base_model = create_advanced_model(
        num_classes=num_classes,
        dropout_rate=0.5,
        l2_reg=0.01
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks_p1 = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    history1 = model.fit(
        train_gen,
        epochs=15,
        validation_data=val_gen,
        callbacks=callbacks_p1,
        verbose=1
    )
    
    # ==========================================
    # PHASE 2: Fine-tune with unfrozen layers
    # ==========================================
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning (unfreezing top layers)")
    print("=" * 60)
    
    # Unfreeze last 20 layers of base model
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks_p2 = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(output_dir, 'hair_type_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    history2 = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks_p2,
        verbose=1
    )
    
    # ==========================================
    # Evaluation
    # ==========================================
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    # Evaluate on validation set
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    
    print(f"\nFinal Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"Final Validation Loss: {val_loss:.4f}")
    
    # Check for overfitting
    train_loss, train_acc = model.evaluate(train_gen, verbose=0)
    print(f"\nTraining Accuracy: {train_acc * 100:.2f}%")
    print(f"Training Loss: {train_loss:.4f}")
    
    gap = train_acc - val_acc
    if gap > 0.1:
        print(f"\n⚠️ Warning: Possible overfitting! Gap: {gap*100:.1f}%")
    elif gap < -0.05:
        print(f"\n⚠️ Warning: Possible underfitting! Gap: {gap*100:.1f}%")  
    else:
        print(f"\n✅ Good fit! Train-Val gap: {gap*100:.1f}%")
    
    # Save class mapping
    with open(os.path.join(output_dir, 'class_mapping.txt'), 'w') as f:
        for class_name, idx in train_gen.class_indices.items():
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
    
    model = train_advanced(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        epochs=40,
        batch_size=16
    )
