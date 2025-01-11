import sys
import tensorflow as tf

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

# Pre-trained BERT model we use for predictions
MODEL = "bert-base-uncased"

# Number of predictions to show for the masked word
K = 3

# Constants to create attention visuals
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40  # Size of grid cells
PIXELS_PER_WORD = 200  # Space for words in diagram


def main():
    text = input("Text: ")  # Prompt user for input text

    # Tokenize the input text
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="tf")
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:  # Check if [MASK] is present
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # Load BERT model and process the input
    model = TFBertForMaskedLM.from_pretrained(MODEL)
    result = model(**inputs, output_attentions=True)

    # Get predictions for the masked word
    mask_token_logits = result.logits[0, mask_token_index]
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    for token in top_tokens:
        # Replace [MASK] with predictions and print each option
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

    # Generate attention diagrams
    visualize_attentions(inputs.tokens(), result.attentions)


def get_mask_token_index(mask_token_id, inputs):
    """
    Find the position of the [MASK] token in the input.
    Returns its index, or None if not found.
    """
    for i, token in enumerate(inputs.input_ids[0]):  # Loop through input IDs
        if token == mask_token_id:
            return i  # Return index of the mask
    return None  # No mask found


def get_color_for_attention_score(attention_score):
    """
    Convert attention score to a shade of gray.
    Darker = lower score, Lighter = higher score.
    """
    attention_score = attention_score.numpy()
    shade = round(attention_score * 255)  # Scale score to [0, 255]
    return (shade, shade, shade)  # RGB triplet for gray shade


def visualize_attentions(tokens, attentions):
    """
    Create attention diagrams for every attention head in all layers.
    Each diagram will be saved as an image file.
    """
    for i, layer in enumerate(attentions):  # Loop through layers
        for k in range(len(layer[0])):  # Loop through attention heads
            layer_number = i + 1  # Start counting from 1
            head_number = k + 1  # Start counting from 1
            generate_diagram(
                layer_number,
                head_number,
                tokens,
                attentions[i][0][k]  # Attention scores for this head
            )


def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Create a visual representation of attention weights for one head.
    Each cell shows how much one token attends to another.
    """
    # Calculate image size based on number of tokens
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")  # Create blank image
    draw = ImageDraw.Draw(img)

    # Draw tokens along top and side of grid
    for i, token in enumerate(tokens):
        # Token columns
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )
        token_image = token_image.rotate(90)  # Rotate token for vertical axis
        img.paste(token_image, mask=token_image)

        # Token rows
        _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )

    # Draw attention weights as shaded grid cells
    for i in range(len(tokens)):
        y = PIXELS_PER_WORD + i * GRID_SIZE
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i][j])
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    # Save the image with layer and head info
    img.save(f"Attention_Layer{layer_number}_Head{head_number}.png")


if __name__ == "__main__":
    main()
