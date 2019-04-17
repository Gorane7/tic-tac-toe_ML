import random
import numpy

do_random = True

class Tic_tac_toe():
	def __init__(self):
		self.board = numpy.zeros(9)
		self.turn = 1
		self.records = numpy.zeros(1)
		self.game_over = False
		self.inputs = []
	def make_move(self, location):
		if self.board[location] == 0:
			self.board[location] = self.turn
			self.records = numpy.append(self.records, location)
			self.check_victory()
			self.next_turn()
			return True
		else:
			self.records = numpy.append(self.records, location)
			self.next_turn()
			self.records[0] = self.turn
			self.game_over = True
			return False
	def check_victory(self):
		if self.check_rows() or self.check_columns() or self.check_diagonals():
			print(str(self.turn) + " has won!!!")
			self.records[0] = self.turn
			self.game_over = True
		elif numpy.all(self.board):
			self.game_over = True
			print("Tie!!!")
	def check_rows(self):
		for i in range(3):
			if self.board[i*3] == self.board[i*3 + 1] == self.board[i*3 + 2] and self.board[i*3] != 0:
				return True
		return False
	def check_columns(self):
		for i in range(3):
			if self.board[i] == self.board[3 + i] == self.board[6 + i] and self.board[i] != 0:
				return True
		return False
	def check_diagonals(self):
		if self.board[0] == self.board[4] == self.board[8] and self.board[0] != 0:
			return True
		if self.board[2] == self.board[4] == self.board[6] and self.board[2] != 0:
			return True
		return False
	def next_turn(self):
		self.turn = self.turn % 2
		self.turn += 1
	def tell_state(self):
		first_perspective = numpy.zeros(9)
		numpy.copyto(first_perspective, self.board)
		first_perspective[first_perspective!=self.turn] = 0
		self.next_turn()
		second_perspective = numpy.zeros(9)
		numpy.copyto(second_perspective, self.board)
		second_perspective[second_perspective!=self.turn] = 0
		self.next_turn()
		board_state = numpy.concatenate((first_perspective, second_perspective))
		board_state[board_state==2] = 1
		self.inputs.append(board_state)
		return board_state
class Network():
	def __init__(self, middle_layer_neuron_amount):
		self.first_layer = numpy.random.rand(18, middle_layer_neuron_amount) * 2.0 - 1.0
		self.second_layer = numpy.random.rand(middle_layer_neuron_amount, 9) * 2.0 - 1.0
		self.first_bias = numpy.random.rand(1,middle_layer_neuron_amount) * 2.0 - 1.0
		self.second_bias = numpy.random.rand(1,9) * 2.0 - 1.0
		self.guesses = []
		self.answers = []
	def forward_pass(self, inputs):
		middle_layer_values = self.sigmoid((inputs @ self.first_layer + self.first_bias))
		output_values = self.sigmoid((middle_layer_values @ self.second_layer + self.second_bias))
		return output_values
	def sigmoid(self, input):
		return 1 / (1 + numpy.exp(-input))
	def pick_random(self, probs):
		choice = random.uniform(0,1)
		cumulative_prob = 0.0
		for i, prob in enumerate(probs.T):
			cumulative_prob += prob
			if choice < cumulative_prob:
				return i
	def normalize_outputs(self, outputs):
		return outputs / numpy.sum(outputs)
	def choose(self, inputs):
		probs = self.forward_pass(inputs)
		move = self.pick_random(self.normalize_outputs(probs))
		answer = numpy.zeros(9)
		answer[move] = 1
		self.answers.append(answer)
		self.guesses.append(probs)
		return move
	def save_network(self):
		numpy.savetxt("first_layer.txt", self.first_layer)
		numpy.savetxt("first_bias.txt", self.first_bias)
		numpy.savetxt("second_layer.txt", self.second_layer)
		numpy.savetxt("second_bias.txt", self.second_bias)
	def gen_train_data(self, inputs, winner):
		self.train_data = []
		if winner == 0:
			print("Tie!!!")
			for x, guess, answer in zip(inputs, self.guesses, self.answers):
				self.train_data.append([x, guess, answer])
		else:
			print(str(winner) + " has won!!!")
			for i, (x, guess, answer) in enumerate(zip(inputs, self.guesses, self.answers)):
				if i % 2 != winner:
					self.train_data.append([x, guess, answer])
					self.train_data.append([x, guess, answer])
	def load_network(self):
		self.first_layer = numpy.loadtxt("first_layer.txt")
		self.first_bias = numpy.loadtxt("first_bias.txt")
		self.second_layer = numpy.loadtxt("second_layer.txt")
		self.second_bias = numpy.loadtxt("second_bias.txt")

test_game = Tic_tac_toe()
test_network = Network(14)
test_network.load_network()

while not test_game.game_over and do_random == True:
	#move = int(input("Location:"))
	#print(test_network.forward_pass(test_game.tell_state()))
	move = test_network.choose(test_game.tell_state())
	if test_game.make_move(move):
		print(test_game.board[0:3])
		print(test_game.board[3:6])
		print(test_game.board[6:9])
		print("")

test_network.save_network()
print(test_game.records)
test_network.gen_train_data(test_game.inputs, test_game.records[0])
print(len(test_network.train_data))
print(test_network.train_data)