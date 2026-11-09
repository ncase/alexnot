from typing import Dict, List

def change_background(train_pairs:List[Dict], parent_program_results:List[Dict]) -> set:
    candidates = set()
    for program_result in parent_program_results:
        for pair_index, pair in enumerate(train_pairs):
            # Before
            predicted_scene = program_result['predicted_scenes'][pair_index]
            # After
            output_scene = pair['output']
            # Hey, if the BG changed, suggest changing it!
            if predicted_scene['background']!=output_scene['background']:
                new_color = output_scene['background']
                extra_code = f'change_background(color="{new_color}")'
                candidates.add(f"{program_result['program']}\n{extra_code}".strip())
    return candidates
