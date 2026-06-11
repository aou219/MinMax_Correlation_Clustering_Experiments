import os
from facebook_sampling import load_facebook_circles

FACEBOOK_DIR = "data/facebook"

ego_ids = sorted([
    filename.replace(".circles", "")
    for filename in os.listdir(FACEBOOK_DIR)
    if filename.endswith(".circles")
])

for ego_id in ego_ids:
    circles_file = f"{FACEBOOK_DIR}/{ego_id}.circles"
    circles = load_facebook_circles(circles_file)

    circle_sizes = sorted(
        [len(circle["nodes"]) for circle in circles],
        reverse=True
    )

    print("\n" + "=" * 70)
    print("ego_id:", ego_id)
    print("number of circles:", len(circles))
    print("circle sizes:", circle_sizes)
