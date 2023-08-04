from models.wide_residual_network import create_wide_residual_network

model = create_wide_residual_network((84, 84, 1), nb_classes=10, N=4, k=10, dropout=0.0)

model.summary()
