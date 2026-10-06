import numpy as np

def gaussian_membership(x, center, sigma):
    """Calculate the fuzzy membership grade."""
    return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


def classify_pattern(input_vector, rules):
    """Classify an input pattern using fuzzy Gaussian membership."""
    
    results = []

    for rule in rules:
        # Calculate membership for each feature
        memberships = []

        for value, (center, sigma) in zip(
            input_vector, rule["params"]
        ):
            membership = gaussian_membership(
                value, center, sigma
            )
            memberships.append(membership)

        # Product represents the AND operation
        rule_strength = np.prod(memberships)
        results.append(rule_strength)

    return np.argmax(results), results


# Define fuzzy rules for two classes
pattern_rules = [
    {
        "class": "Type A",
        "params": [(0.5, 0.1), (0.5, 0.1)]
    },
    {
        "class": "Type B",
        "params": [(0.8, 0.2), (0.2, 0.2)]
    }
]


# Input pattern
sample_input = [0.55, 0.45]

# Perform classification
prediction, scores = classify_pattern(
    sample_input, pattern_rules
)

# Display results
print("Membership scores:")
print("Type A:", scores[0])
print("Type B:", scores[1])

print()
print(
    f"The detected pattern is: "
    f"{pattern_rules[prediction]['class']}"
)
