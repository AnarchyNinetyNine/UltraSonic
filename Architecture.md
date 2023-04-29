# Architecture:

1. Input shape: (224, 224, 3)
2. `Conv2D` layer with 8 filters, 3x3 kernel size, `L2` regularization, and `He normal` initializer
3. `ReLU` activation layer
4. `Layer normalization` (axis=1)
5. `Max pooling` layer (pool size: 2x2)

6. `Conv2D` layer with 16 filters, 3x3 kernel size, `L2` regularization, and `He normal` initializer
7. `ReLU` activation layer
8. `Layer normalization` (axis=1)
9. `Max pooling` layer (pool size: 2x2)

10. `Conv2D` layer with 32 filters, 3x3 kernel size, `L2` regularization, and `He normal` initializer
11. `ReLU` activation layer
12. `Layer normalization` (axis=1)
13. `Max pooling` layer (pool size: 2x2)

14. `Conv2D` layer with 64 filters, 3x3 kernel size, `L2` regularization, and `He normal` initializer
15. `ReLU` activation layer
16. `Layer normalization` (axis=1)
17. `Max pooling` layer (pool size: 2x2)

18. `Conv2D` layer with 128 filters, 3x3 kernel size, `L2` regularization, and `He normal` initializer
19. `ReLU` activation layer
20. `Layer normalization` (axis=1)
21. `Max pooling` layer (pool size: 2x2)

22. `Dropout` layer (dropout rate: 0.2)
23. `Global average pooling` layer

24. `Dense` layer with 128 units and `He normal` initializer
25. `ReLU` activation layer
26. `Layer normalization` (axis=1)
27. `Dropout` layer (dropout rate: 0.2)

28. `Dense` layer with 64 units and `He normal` initializer
29. `ReLU` activation layer
30. `Layer normalization` (axis=1)
31. `Dropout` layer (dropout rate: 0.2)

32. `Dense` layer with 1 unit and `He normal` initializer
33. `Sigmoid` activation layer

# Callbacks:
- `EarlyStopping`: monitor='val_loss', patience=5, verbose=1
- `ModelCheckpoint`: 'kernel.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1
- `ReduceLROnPlateau`: monitor='val_loss', factor=0.2, patience=3, verbose=1, min_lr=1e-6
