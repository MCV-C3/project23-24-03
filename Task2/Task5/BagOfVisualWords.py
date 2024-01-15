import numpy as np
import pickle
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import optuna


class BagOfVisualWords:
    def __init__(self,
                 codebook, codebook_parameters,  # Codebook types K-MEANS, FISHER
                 classifier, classifier_parameters,  # Classifier types KNN, SVC, LogisticRegression
                 scaler,  # Scaler types None, MinMax, Stardarize, Normalize
                 dim_red, dim_red_parameters,  # Dimension Reduction None, PCA, LDA
                 ):

        # Initialize codebook
        if codebook == 'K-MEANS':
            self.codebook = MiniBatchKMeans(**codebook_parameters)

        # Initialize the classifier
        if classifier == 'KNN':
            self.classifier = KNeighborsClassifier(**classifier_parameters)

        elif classifier == 'SVC':
            self.classifier = SVC(**classifier_parameters)

        elif classifier == 'LOGISTICREGRESSION':
            self.classifier = LogisticRegression(**classifier_parameters)

        # Initialize scaler
        self.scaler = None
        if scaler == 'MINMAX':
            self.scaler = MinMaxScaler()

        elif scaler == 'STANDARIZE':
            self.scaler = StandardScaler()

        elif scaler == 'NORMALIZE':
            self.scaler = Normalizer()

        # Initalize dimension reduction
        self.dim_red = None

        if dim_red == 'PCA':
            self.dim_red = PCA(**dim_red_parameters)

        self.codebook_used = codebook

    def get_k(self):
        if self.codebook_used == 'K-MEANS':
            return len(self.codebook.cluster_centers_)

    def fit(self, X_train, y_train):
        X_train_column = np.vstack(X_train)

        # 2. Train the codebook
        self.codebook.fit(X_train_column)

        # 3. Compute the words on each image
        k = self.get_k()
        visual_words = np.zeros((len(X_train), k), dtype=np.float32)
        for i in range(len(X_train)):  # [[[], ..., []], []]
            words = self.codebook.predict(X_train[i])
            visual_words[i, :] = np.bincount(words, minlength=k)

        # 4. Use the scaler
        if self.scaler is not None:
            self.scaler.fit(visual_words)
            visual_words = self.scaler.transform(visual_words)

        # 4. Use the dimension reduction
        if self.dim_red is not None:
            self.dim_red.fit(visual_words)
            visual_words = self.dim_red.transform(visual_words)

        # 5. Train the classifier
        self.classifier.fit(visual_words, y_train)

    def predict(self, X):
        # 1. Create the features
        k = self.get_k()
        visual_words = np.zeros((len(X), k), dtype=np.float32)
        for i in range(len(X)):
            words = self.codebook.predict(X[i])
            visual_words[i, :] = np.bincount(words, minlength=k)

        # 3. Use the scaler
        if self.scaler is not None:
            visual_words = self.scaler.transform(visual_words)

        # 4. Use the dimension reduction
        if self.dim_red is not None:
            visual_words = self.dim_red.transform(visual_words)

        # 5. Train the classifier
        return self.classifier.predict(visual_words)

    def score(self, X, y_true):
        # 1. Create the features
        k = self.get_k()
        visual_words = np.zeros((len(X), k), dtype=np.float32)
        for i in range(len(X)):
            words = self.codebook.predict(X[i])
            visual_words[i, :] = np.bincount(words, minlength=k)

        # 3. Use the scaler
        if self.scaler is not None:
            visual_words = self.scaler.transform(visual_words)

        # 4. Use the dimension reduction
        if self.dim_red is not None:
            visual_words = self.dim_red.transform(visual_words)

        # 5. Train the classifier
        return 100 * self.classifier.score(visual_words, y_true)


def load_files(train_images_file, test_images_file, train_labels_filename, test_labels_filename):
    train_images_filenames = pickle.load(open(train_images_file, 'rb'))
    test_images_filenames = pickle.load(open(test_images_file, 'rb'))
    train_images_filenames = ['..' + n[15:] for n in train_images_filenames]
    test_images_filenames = ['..' + n[15:] for n in test_images_filenames]
    train_labels = pickle.load(open(train_labels_filename, 'rb'))
    test_labels = pickle.load(open(test_labels_filename, 'rb'))

    return train_images_filenames, test_images_filenames, train_labels, test_labels

def objective(trial):
        BOVWSample = BagOfVisualWords(
                                    codebook = 'K-MEANS', # Codebook types K-MEANS, FISHER
                                    codebook_parameters = {
                                            'n_clusters': trial.suggest_int('n_clusters', 16, 256, step=8),
                                            'verbose': False,
                                            'batch_size': trial.suggest_int('batch_size', 128, 256, step=4) * 20,
                                            'compute_labels': False,
                                            'reassignment_ratio': 10**-4,
                                            'random_state': 42
                                        },
                                    classifier = 'KNN', # Classifier types KNN, SVC, LOGISTICREGRESSION
                                    classifier_parameters = {
                                            'n_neighbors': trial.suggest_int('n_neighbors', 2, 60, step=2),
                                            'n_jobs': -1,
                                            'metric': trial.suggest_categorical('metric', ['euclidean', 'cosine'])
                                        },
                                    scaler = trial.suggest_categorical('scaler', ['MINMAX', 'STANDARIZE', 'NORMALIZE']), # Scaler types None, MinMax, Stardarize, Normalize
                                    dim_red = trial.suggest_int('n_dims', 64, 256, step=4),
                                    dim_red_parameters = trial.suggest_int('n_dims', 64, 256, step=4), # Dimension Reduction None, PCA
                                    )
        
        BOVWSample.fit(train_data, train_labels)
        accuracy = BOVWSample.score(test_data, test_labels)
        return accuracy
    
def create_batches(input_list, batch_size):
    batches = []
    for i in range(0, len(input_list), batch_size):
        batch = input_list[i:i + batch_size]
        batches.append(batch)
    return batches

if __name__ == '__main__':
    with open('train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
        train_data = np.array(train_data)
    
    with open('train_labels.pkl', 'rb') as f:
        train_labels = pickle.load(f)

    with open('test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
        test_data = np.array(test_data)
    
    with open('test_labels.pkl', 'rb') as f:
        test_labels = pickle.load(f)
        
    print(train_data.shape, test_data.shape, len(train_labels), len(test_labels))
    
    study = optuna.create_study(storage='sqlite:///optunaTask2BoW.db', study_name= "task2_v1", direction='maximize')  # Create a new study.
    study.optimize(objective, n_trials=200)
    
                            