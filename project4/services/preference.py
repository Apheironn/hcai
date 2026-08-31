import numpy as np


def _log_sum_exp(values):
    shift = values.max()
    return shift + np.log(np.exp(values - shift).sum())


def _softmax(values):
    shifted = values - values.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


class PreferenceModel:
    """MAP estimate of the latent preference vector w for U(x) = w^T x.

    Rankings i1 > i2 > ... > in are modelled with the Plackett-Luce extension of
    Bradley-Terry: the ranking is the product of the choice probabilities of each
    item over the items still available. With n = 2 this reduces exactly to the
    standard Bradley-Terry pairwise model.
    """

    def __init__(self, weights, n_rankings=0, log_posterior=0.0):
        self.weights = weights
        self.n_rankings = n_rankings
        self.log_posterior = log_posterior

    @classmethod
    def fit(cls, features, rankings, l2=1.0, lr=0.5, n_iter=400, tol=1e-7):
        """Gradient ascent on the log posterior (Plackett-Luce likelihood + Gaussian prior)."""
        weights = np.zeros(features.shape[1])
        blocks = [features[list(ranking)] for ranking in rankings if len(ranking) >= 2]
        if not blocks:
            return cls(weights)

        value = cls._log_posterior(blocks, weights, l2)
        for _ in range(n_iter):
            gradient = cls._gradient(blocks, weights, l2)
            step = lr
            for _ in range(20):
                candidate = weights + step * gradient
                candidate_value = cls._log_posterior(blocks, candidate, l2)
                if candidate_value >= value:
                    break
                step /= 2.0
            else:
                break
            if candidate_value - value < tol:
                weights, value = candidate, candidate_value
                break
            weights, value = candidate, candidate_value

        return cls(weights, n_rankings=len(blocks), log_posterior=float(value))

    @staticmethod
    def _log_posterior(blocks, weights, l2):
        total = -0.5 * l2 * float(weights @ weights)
        for block in blocks:
            utilities = block @ weights
            for k in range(len(utilities) - 1):
                total += utilities[k] - _log_sum_exp(utilities[k:])
        return total

    @staticmethod
    def _gradient(blocks, weights, l2):
        gradient = -l2 * weights
        for block in blocks:
            utilities = block @ weights
            for k in range(len(utilities) - 1):
                probabilities = _softmax(utilities[k:])
                gradient += block[k] - probabilities @ block[k:]
        return gradient

    def utilities(self, features):
        return features @ self.weights

    def pair_probability(self, features, winner, loser):
        """Bradley-Terry probability that `winner` is preferred to `loser`."""
        gap = float(features[winner] @ self.weights - features[loser] @ self.weights)
        return 1.0 / (1.0 + np.exp(-gap))

    def pair_accuracy(self, features, pairs):
        """Share of held-out pairwise judgements (winner, loser) predicted correctly."""
        if not pairs:
            return None
        utilities = self.utilities(features)
        correct = sum(1 for winner, loser in pairs if utilities[winner] > utilities[loser])
        return round(correct / len(pairs), 4)

    def recommend(self, features, k=5, exclude=()):
        utilities = self.utilities(features)
        order = np.argsort(-utilities)
        excluded = set(exclude)
        return [int(i) for i in order if int(i) not in excluded][:k]

    def weight_table(self, feature_names, k=6):
        pairs = sorted(zip(feature_names, self.weights), key=lambda item: -abs(item[1]))
        return [{"name": name, "weight": round(float(value), 3)} for name, value in pairs[:k]]

    @staticmethod
    def implied_pairs(ranking):
        """All pairwise judgements contained in a ranking (used only for reporting)."""
        return [(ranking[i], ranking[j]) for i in range(len(ranking)) for j in range(i + 1, len(ranking))]
