import torch
import click
import tqdm
import matplotlib.pyplot as plt

# import from the other scripts
from data import mnist
from model import MyAwesomeModel


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()
    
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    train_losses = []
    for e in tqdm.tqdm(range(epochs),unit='epoch'):
        running_loss = 0
        for images, labels in trainloader:
            # perform training
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            # update the loss
            running_loss += loss.item()
        train_losses.append(running_loss) # append

    # save the model
    torch.save(model.state_dict(), 'trained_model.pth')
    # plot the losses
    plt.plot(train_losses)
    plt.show()

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist()
    
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
    results = []
    with torch.no_grad():
        model.eval()
        for images, labels in testloader:
            # produce predictions
            log_ps = model(images)
            ps = torch.exp(log_ps)
            _, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            results.append(equals)


        equals = torch.cat(results)
        accuracy = torch.mean(equals.type(torch.FloatTensor))
        print(f'Accuracy: {accuracy.item()*100}%')


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
