#include <cmath>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

class NaiveBayes {
public:
  NaiveBayes();
  ~NaiveBayes();

  void train(const std::vector<std::string> &data,
             const std::vector<int> &labels);
  int predict(const std::string &data);

private:
  std::map<std::string, int> word_counts;
  std::map<int, std::map<std::string, int>> label_word_counts;
  std::map<int, int> label_counts;
};

NaiveBayes::NaiveBayes() {
  // Initialization code
}

NaiveBayes::~NaiveBayes() {
  // Cleanup code
}

void NaiveBayes::train(const std::vector<std::string> &data,
                       const std::vector<int> &labels) {

  for (size_t i = 0; i < data.size(); i++) {
    std::string line = data[i];
    int label = labels[i];

    label_counts[label]++;

    std::stringstream ss(line);

    std::string word;
    while (ss >> word) {
      word_counts[word]++;
      label_word_counts[label][word]++;
    }
  }
}

int NaiveBayes::predict(const std::string &data) {
  int best_label = -1;
  double best_prob = -INFINITY;

  for (const auto &label_entry : label_counts) {
    int label = label_entry.first;
    double prob = log(double(label_entry.second) / data.size()); // P(label)

    // std::cout << "Initial Prob for label " << label << ": " << prob
    //           << std::endl; // Debugging line

    std::stringstream ss(data);
    std::string word;
    while (ss >> word) {
      double word_prob =
          log(double(label_word_counts[label][word] + 1) /
              (label_entry.second + word_counts.size())); // P(word|label)
      prob += word_prob;

      // std::cout << "Word: " << word << ", Prob: " << word_prob
      //           << std::endl; // Debugging line
    }

    // std::cout << "Final Prob for label " << label << ": " << prob
    //           << std::endl; // Debugging line

    if (prob > best_prob) {
      best_label = label;
      best_prob = prob;
    }
  }

  return best_label;
}

int main() {
  NaiveBayes nb;

  std::vector<std::string> train_data = {"buy now",
                                         "limited offer",
                                         "special deal",
                                         "exclusive offer",
                                         "win money",
                                         "urgent",
                                         "hello",
                                         "how are you",
                                         "good morning",
                                         "nice to meet you",
                                         "long time no see",
                                         "how have you been"};
  std::vector<int> train_labels = {1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0};
  nb.train(train_data, train_labels);

  std::vector<std::string> test_data = {
      "buy offer",   "how now",      "hello you",      "subscribe now",
      "urgent sale", "good evening", "how's it going", "long time"};

  for (const auto &data : test_data) {
    int prediction = nb.predict(data);
    std::cout << "Predicted label for \"" << data << "\": " << prediction
              << std::endl;
  }

  return 0;
}