from matplotlib import pyplot as plt

from point_cloud_diff.difftda import *
from gudhi.wasserstein import wasserstein_distance


np.random.seed(1)
angles = np.random.uniform(0, 2 * np.pi, 100)
X = np.array([[0.1, 0.], [1.5, 1.5], [0., 1.6]])
dim = 0


XTF = tf.Variable(X, tf.float32)
lr = 1
optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

num_epochs = 1
losses, Dgs, Xs, grads = [], [], [], []
for epoch in range(num_epochs + 1):
    with tf.GradientTape() as tape:
        Dg = RipsModel(X=XTF, mel=10, dim=dim, card=10).call()
        loss = -wasserstein_distance(Dg, tf.constant(np.empty([0, 2])), order=1, enable_autodiff=True)

    Dgs.append(Dg.numpy())
    Xs.append(XTF.numpy())
    losses.append(loss.numpy())

    gradients = tape.gradient(loss, [XTF])
    grads.append(gradients[0].numpy())
    optimizer.apply_gradients(zip(gradients, [XTF]))


pts_to_move = np.argwhere(np.linalg.norm(grads[0], axis=1) != 0).ravel()
plt.figure()
for pt in pts_to_move:
    plt.arrow(Xs[0][pt,0], Xs[0][pt,1], -lr*grads[0][pt,0], -lr*grads[0][pt,1], color='blue',
              length_includes_head=True, head_length=.05, head_width=.1, zorder=10)
plt.scatter(Xs[0][:,0], Xs[0][:,1], c='red', s=50, alpha=.2,  zorder=3)
plt.scatter(Xs[0][pts_to_move,0], Xs[0][pts_to_move,1], c='red',   s=150, marker='o', zorder=2, alpha=.7, label='Step i')
plt.scatter(Xs[1][pts_to_move,0], Xs[1][pts_to_move,1], c='green', s=150, marker='o', zorder=1, alpha=.7, label='Step i+1')
plt.axis('square')
plt.legend()
plt.show()
