import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

def augment_scrapable():
    print("Loading labels.csv...")
    df = pd.read_csv('data/labels.csv')
    
    # Filter scrapable
    scrapable_df = df[df['label'] == 1]
    print(f"Found {len(scrapable_df)} Scrapable images.")
    
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    new_rows = []
    
    # We will generate 4 augmentations per image
    aug_factor = 4
    
    max_id = df['image_id'].max()
    print(f"Generating {aug_factor} augmentations per Scrapable image...")
    
    import uuid
    
    for index, row in scrapable_df.iterrows():
        img_path = os.path.join('data', 'images', str(row['filename']))
        if not os.path.exists(img_path):
            continue
            
        img = load_img(img_path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        
        i = 0
        for batch in datagen.flow(x, batch_size=1):
            new_img = array_to_img(batch[0])
            # Save the new image
            new_filename = f"aug_{uuid.uuid4().hex[:8]}.jpg"
            new_img_path = os.path.join('data', 'images', new_filename)
            new_img.save(new_img_path)
            
            max_id += 1
            new_rows.append({'image_id': max_id, 'filename': new_filename, 'label': 1})
            
            i += 1
            if i >= aug_factor:
                break
                
    new_df = pd.DataFrame(new_rows)
    df_combined = pd.concat([df, new_df], ignore_index=True)
    df_combined.to_csv('data/labels.csv', index=False)
    
    print(f"Successfully generated {len(new_rows)} new Scrapable images.")
    print("New distribution:")
    print(df_combined['label'].value_counts())

if __name__ == '__main__':
    augment_scrapable()
