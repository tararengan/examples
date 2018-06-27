#A binary classifier trained on a neural network using pytorch


#imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import math
import sys



def generate_rect_points(size = 50):

    N = size
    c1_points = np.multiply(np.random.choice([1, -1], size=(N, 1)),
                         np.random.randint(4, 10, (N, 1), 'int32')  + np.random.randn(N, 1))
    c2_points = np.multiply(np.random.choice([1, -1], size=(N, 1)),
                            np.random.randint(3, 10, (N, 1), 'int32') + np.random.randn(N, 1))

    points = np.concatenate((c1_points, c2_points), axis=1).reshape(N, 2)

    return points


def generate_points_on_circle(size=50, num_classes=4):

    radius_multiplier = np.random.choice(list(np.array(range(num_classes))+1), size=(size,))
    c1_points = np.multiply(radius_multiplier, np.random.random_sample(size))
    c2_points = np.sqrt(np.square(radius_multiplier) - np.square(c1_points))

    points = np.stack((c1_points, c2_points), axis=-1)

    return points, radius_multiplier


def generate_labels_by_quad(points):
    N = points.shape[0]
    labels = np.ones((N,))

    c1_points =  points[:, 0].reshape(N,)
    c2_points = points[:, 1].reshape(N,)

    indices_q1 = np.where((c1_points > 0) & (c2_points > 0))
    indices_q2 = np.where((c1_points <= 0) & (c2_points > 0))
    indices_q3 = np.where((c1_points <= 0) & (c2_points <= 0))
    indices_q4 = np.where((c1_points > 0) & (c2_points <= 0))

    labels[indices_q1[0]] = 0
    labels[indices_q2[0]] = 1
    labels[indices_q3[0]] = 2
    labels[indices_q4[0]] = 3

    x_ = points
    y_ = labels

    x_ = torch.from_numpy(x_).float()
    y_ = torch.from_numpy(y_).long()

    return x_, y_


def generate_labels_by_rot_quad(points):
    N = points.shape[0]
    labels = np.ones((N,))

    c1_points = points[:, 0].reshape(N, 1)
    c2_points = points[:, 1].reshape(N, 1)

    indices_q1 = np.where((c2_points > 0) & (abs(c1_points) < abs(c2_points)))
    indices_q2 = np.where((c1_points <= 0) & (abs(c2_points) <= abs(c1_points)))
    indices_q3 = np.where((c2_points < 0) & (abs(c1_points) < abs(c2_points)))
    indices_q4 = np.where((c1_points > 0) & (abs(c2_points) <= abs(c1_points)))

    labels[indices_q1[0]] = 0
    labels[indices_q2[0]] = 1
    labels[indices_q3[0]] = 2
    labels[indices_q4[0]] = 3

    x_ = points
    y_ = labels

    x_ = torch.from_numpy(x_).float()
    y_ = torch.from_numpy(y_).long()

    # print(type(y), y.size(), y.dtype)

    return x_, y_


def generate_labels_by_rad(points, radius_mult):
    radius_values = np.unique(radius_mult)
    radius_label_dict = dict([(val, i) for i, val in enumerate(radius_values)])

    labels = np.array(list(map(lambda arg_x: radius_label_dict[arg_x], radius_mult)))

    x_ = points
    y_ = labels

    x_ = torch.from_numpy(x_).float()
    y_ = torch.from_numpy(y_).long()

    return x_, y_


def generate_rect_data(batch_size =50):
    points = generate_rect_points(batch_size)
    x, y = generate_labels_by_rot_quad(points)

    return x, y


def generate_circ_data(batch_size =50, num_classes =4):
    points, radius_multiplier = generate_points_on_circle(batch_size, num_classes)
    x, y = generate_labels_by_rad(points, radius_multiplier)

    return x, y


def get_batch(batch_size=50, num_classes =4):
    #x, y = generate_rect_data(batch_size)
    x, y = generate_circ_data(batch_size, num_classes)
    return x, y


#define model - takes in features and returns log probabilities
class LinearNet(nn.Module):

    def __init__(self):
        super(LinearNet, self).__init__()
        self.lin1 = nn.Linear(2, 2)
        self.lin2 = nn.Linear(2, 4)
        self.activation = nn.ReLU()
        self.logprob = nn.LogSoftmax()

    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.activation(x).squeeze()
        #x = self.logprob(x)
        return x


class CircularNet(nn.Module):

    def __init__(self, num_classes=4):
        super(CircularNet, self).__init__()
        self.lin = nn.Linear(3, num_classes, bias = False)
        self.activation = nn.Hardtanh(-1.5, 1.5)
        self.logprob = nn.LogSoftmax()

    def forward(self, x):
        x = self.lin(x)
        # x = self.activation(x).squeeze()
        #x = self.logprob(x)
        return x


def train_linear(num_batches, x_data, y_data, net, loss_fn, optimizer):

    predictions = []
    for epoch in range(40):

        running_loss = 0.0

        for i in range(num_batches):

            inputs = x_data[i]
            labels = y_data[i]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:  # print every r mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0

    print('Finished training \n')

    return net


def train_circular(num_batches, circ_data, y_data, net, loss_fn, optimizer):

    for epoch in range(50):

        running_loss = 0.0

        for i in range(num_batches):

            inputs = circ_data[i]
            labels = y_data[i]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:  # print every r mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0

    print('Finished training \n')

    return net


def get_accuracy_train(num_batches, num_classes, input_data, x_data, y_data, net, toplot=False):
    if toplot:
        plt.title('A 2D Classifier')
        plt.ion()
        plt.show()

    #get predictions and accuracy
    accuracy = 0
    coord_values_by_class = []
    for i in range(num_batches):

        inputs = input_data[i]
        labels = y_data[i]

        outputs = net(inputs)
        max_values, predictions = torch.max(outputs, 1)

        accuracy += len(np.where(labels.numpy() == predictions.numpy())[0])

        indices_by_class = []
        for j in range(num_classes):
            indices_by_class.append(list(np.where(predictions.numpy() == j)[0]))
            # print(j, outputs, predictions, np.where(predictions.numpy() == j)[0])

        coords_by_class = []
        for j in range(num_classes):
            coords_by_class.append(x_data[i].numpy()[indices_by_class[j], :])

        #print(coords_by_class)

        if i == 0:
            for j in range(num_classes):
                coord_values_by_class.append(coords_by_class[j])
        else:
            for j in range(num_classes):
                coord_values_by_class[j] = np.concatenate(
                    (coord_values_by_class[j], coords_by_class[j]), axis=0)

    accuracy = accuracy/(batch_size*num_batches)



    marker_size = 5
    if toplot:
        for j in range(num_classes):
            plt.scatter(coord_values_by_class[j][:, 0], coord_values_by_class[j][:, 1], s=marker_size)
            marker_size += 20

    return accuracy



# net = LinearNet()


#get data in batches
x_data = []
y_data = []
num_batches = 10
batch_size = 30
num_classes = 5
for i in range(num_batches):
    x, y = get_batch(batch_size, num_classes)
    x_data.append(x)
    y_data.append(y)


net = CircularNet(num_classes)

theta_data = []
circ_data = []
for i in range(len(x_data)):
    rho_data = np.sqrt(np.sum(list(map(lambda arg_x: arg_x**2, x_data[i].numpy())), axis=1))
    rho_sq_data = np.square(rho_data)
    vec_data = np.stack((rho_sq_data, rho_data, np.ones(len(rho_data),)), axis=-1)
    circ_data.append(torch.from_numpy(vec_data))
    circ_data[i] = circ_data[i].float()
    y_data[i] = y_data[i].long()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=.003, momentum=.8)

# start_weight = torch.from_numpy(np.array([[-1, 2, -1], [-1, 4, -4], [-1, 6, -9], [-1, 8, -16], [-1, 10, -25]], dtype=float))
# net.lin.weight.data = start_weight.float()


net = train_circular(num_batches, circ_data, y_data, net, loss_fn, optimizer)
accuracy = get_accuracy_train(num_batches, num_classes, circ_data, x_data, y_data, net, True)

print('Accuracy: {0}'.format(accuracy))

print(net.lin.weight.data)





plt.pause(.001)
input('Press enter to continue')



















