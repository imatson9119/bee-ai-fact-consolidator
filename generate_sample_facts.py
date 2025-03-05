#!/usr/bin/env python3
"""
Generate sample facts for testing the fact consolidator.
This script exports facts to a JSON file that can be used for testing.
"""

import json
import argparse

# Sample facts organized by topic
SAMPLE_FACTS = {
    "friends": [
        "Ian is friends with someone named Scott",
        "Ian has a friend named Scott",
        "Scott is friends with Ian",
        "Ian's friend Scott is the best man at his wedding",
        "Ian met Scott in college",
        "Scott and Ian have been friends for over 10 years"
    ],
    "location": [
        "Ian lives in Seattle",
        "Ian is from Seattle, WA",
        "Ian is a resident of Seattle",
        "Ian moved to Seattle after college",
        "Ian has lived in Seattle for several years"
    ],
    "coffee": [
        "Ian loves coffee",
        "Ian enjoys drinking coffee",
        "Ian is a coffee enthusiast",
        "Ian prefers dark roast coffee",
        "Ian visits coffee shops regularly"
    ],
    "career": [
        "Ian works as a software engineer",
        "Ian's job is software engineering",
        "Ian is a programmer by profession",
        "Ian develops software for a living",
        "Ian has been a software engineer for many years"
    ],
    "hobbies": [
        "Ian plays the guitar",
        "Ian knows how to play guitar",
        "Ian enjoys playing guitar in his free time",
        "Ian has been playing guitar since high school",
        "Ian owns an acoustic guitar"
    ],
    "travel": [
        "Ian traveled to Japan last year",
        "Ian visited Tokyo in 2022",
        "Ian went on a trip to Japan",
        "Ian enjoyed his vacation in Japan",
        "Ian explored several cities in Japan during his trip"
    ]
}


def generate_facts(num_facts=30, output_file="sample_facts.json"):
    """
    Generate sample facts by selecting from the available topics.
    Ensures a mix of similar facts that can be consolidated.
    """
    import random
    
    # Select facts
    selected_facts = []
    topics = list(SAMPLE_FACTS.keys())
    
    while len(selected_facts) < num_facts:
        # Pick a random topic
        topic = random.choice(topics)
        
        # Pick a random fact from that topic
        facts = SAMPLE_FACTS[topic]
        fact = random.choice(facts)
        
        # Add it if not already selected
        if fact not in selected_facts:
            selected_facts.append(fact)
    
    # Convert to fact objects with IDs
    fact_objects = [{"id": i+1, "text": fact} for i, fact in enumerate(selected_facts)]
    
    # Write to file
    with open(output_file, 'w') as f:
        json.dump(fact_objects, f, indent=2)
    
    print(f"Generated {len(fact_objects)} sample facts and saved to {output_file}")
    return fact_objects


def main():
    parser = argparse.ArgumentParser(description="Generate sample facts for testing")
    parser.add_argument("--num", type=int, default=30, help="Number of facts to generate")
    parser.add_argument("--output", type=str, default="sample_facts.json", help="Output JSON file")
    args = parser.parse_args()
    
    generate_facts(args.num, args.output)


if __name__ == "__main__":
    main() 