from magenta.models.transformer import melody_rnn_sequence_generator
from magenta.models.shared import sequence_generator_bundle
import note_seq
from note_seq.protobuf import music_pb2
import tempfile
import os

# Text ko seed melody mein convert karna
def text_to_seed_melody(text):
    # Simple mapping text se notes mein
    notes = []
    for char in text:
        pitch = (ord(char) % 12) + 60  # ASCII values se pitch create karna
        note = music_pb2.NoteSequence.Note(
            pitch=pitch, start_time=len(notes), end_time=len(notes) + 1, velocity=80
        )
        notes.append(note)
    return notes

# Music generate karne ka function
def generate_music_from_text(text, output_file='output.mid'):
    # Text ko seed melody mein badle
    seed_melody = text_to_seed_melody(text)
    
    # Seed melody ko NoteSequence mein badlein
    sequence = music_pb2.NoteSequence(notes=seed_melody)
    
    # Magenta Melody RNN Model ka bundle load karein
    bundle = sequence_generator_bundle.read_bundle_file(
        'https://storage.googleapis.com/magentadata/models/melody_rnn/basic_rnn.mag'
    )
    generator = melody_rnn_sequence_generator.MelodyRnnSequenceGenerator(
        checkpoint=None, bundle=bundle
    )
    
    # Model configuration set karein
    generator.initialize()
    generator_options = generator.default_generate_sequence_request()
    generator_options.num_steps = 128  # Notes ki length
    generator_options.temperature = 1.0  # Creativity level
    
    # Music generate karein
    generated_sequence = generator.generate(sequence, generator_options)
    
    # Output MIDI file save karein
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        note_seq.sequence_proto_to_midi_file(generated_sequence, temp_file.name)
        os.rename(temp_file.name, output_file)
    print(f"Music saved to {output_file}")

# Text input aur music generation
text_input = "AI-generated music is amazing!"
generate_music_from_text(text_input)
