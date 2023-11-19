def train_model(model, trainloader, epochs=1):
    model.fit(trainloader, epochs=epochs)
