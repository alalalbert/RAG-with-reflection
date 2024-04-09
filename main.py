import anthropic
import os

client = anthropic.Anthropic(os.environ['ANTHROPIC_API_KEY'])


class Essay:
  def __init__(self, topic, tone, audience, length, key_points, sources):
    self.topic = topic
    self.tone = tone
    self.audience = audience
    self.length = length
    self.key_points = key_points
    self.sources = sources

  def input_for_attr(prompt):
    return input(prompt)

  def from_user_input(cls):
    topic = cls.input_for_attr("What is the topic of your essay? ")
    tone = cls.input_for_attr("What is the tone of your essay? ")
    audience = cls.input_for_attr("Who is the audience of your essay? ")
    length = cls.input_for_attr("What is the length of your essay? ")
    key_points = cls.input_for_attr("What are the key points of your essay? ")
    sources = cls.input_for_attr("What are the sources of your essay? ")
    return cls(topic, tone, audience, length, key_points, sources)

    
  
def generate():
  message = client.messages.create(
  model="claude-3-opus-20240229",
  max_tokens=1000,
  temperature=0.0,
  system="You are a brilliant writing coach. You are exceptionally clear and lucid in your explanations. You make recommendations that are at once thoughtful and creative, and always do your very best work",
  messages=[
      {"role": "user", "content": ""}
  ]
)


def reflect(focus_text):
  if len(focus_text) > 16000: #est of num of chars in 7-page essay, single spaced, per chatGPT (12,250-15,050 est range) Can adjust this to be longer or shorter, based on cost preferences
    return "Sorry, your text is too long for reflection."
  
  message = client.messages.create(
  model="claude-3-opus-20240229",
  max_tokens=1000,
  temperature=0.0,
  system="You are a phenomenal editor of essays. You have years of professional experience in journalism and literary publishing, and it is your greatest joy to edit writers' works to be clear, stylish, and attractive. It is your highest goal to ",
  messages=[
      {"role": "user", "content": [{"type": "text", "text": focus_text]}
  ]
)


