from model_evaluation import ModelEvaluator

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    print(evaluator.evaluate([0.004086771438464067, 17, 8, 40, 0.25, 0.25]))