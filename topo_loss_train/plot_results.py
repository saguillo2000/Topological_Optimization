import pickle

if __name__ == '__main__':
    with open('losses_epochs.pkl', 'rb') as f:
        losses = pickle.load(f)

    print(losses)