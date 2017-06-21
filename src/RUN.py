from DQN_Dueling import DeepQNetwork as Brain
from Atari_Breakout import AtariGame as Game

emulator = Game(_white=True)
agent = Brain()
agent.start_playing(emulator)
