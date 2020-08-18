from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
the_model = 'mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es'
tokenizer = AutoTokenizer.from_pretrained(the_model, do_lower_case=False)
model = AutoModelForQuestionAnswering.from_pretrained(the_model)
from textwrap import wrap

contexto = 'Desde hace un año y medio, las fuerzas federales tendieron un cerco a El Marro, quien desde hace tres años y medio implementó una serie de acciones, como el bloqueo de carreteras y el incendio de vehículos, para evitar su detención. Su grupo mantiene una fuerte disputa con el Cártel Jalisco Nueva Generación, al que le declaró la guerra en 2017; es considerada la organización más poderosa y peligrosa de México, misma que desde hace casi nueve años controla buena parte de Guanajuato, tras quitarle el control a Los Zetas.'
pregunta = '¿quien tenía secuestrada a una persona?'

encode = tokenizer.encode_plus(pregunta, contexto, return_tensors='pt')
input_ids = encode['input_ids'].tolist()
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
for id, token in zip(input_ids[0], tokens):
  print('{:<12} {:>6}'.format(token, id))
  print('')

nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
salida = nlp({'question':pregunta, 'context':contexto})
print(salida)

def pregunta_respuesta(model, contexto, nlp):

  # Imprimir contexto
  print('Contexto:')
  print('-----------------')
  print('\n'.join(wrap(contexto)))

  # Loop preguntas-respuestas:
  continuar = True
  while continuar:
    print('\nPregunta:')
    print('-----------------')
    pregunta = str(input())

    continuar = pregunta!=''

    if continuar:
      salida = nlp({'question':pregunta, 'context':contexto})
      print('\nRespuesta:')
      print('-----------------')
      print(salida['answer'])