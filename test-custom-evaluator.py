from answer_length import AnswerLengthEvaluator

# Instantiate the evaluator
evaluator = AnswerLengthEvaluator()

# Call the evaluator with the answer
answer_length = evaluator(answer="What is the speed of light?")
print(answer_length)