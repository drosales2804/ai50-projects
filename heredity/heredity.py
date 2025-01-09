import csv
import itertools
import sys

PROBS = {

    # Base probabilities for the number of mutated genes
    "gene": {
        2: 0.01,  # 2 copies
        1: 0.03,  # 1 copy
        0: 0.96   # 0 copies
    },

    "trait": {

        # Probability of showing the trait with 2 copies of the gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of showing the trait with 1 copy of the gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of showing the trait with 0 copies of the gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation chance for gene inheritance
    "mutation": 0.01
}


def main():
    """
    Main logic for loading data and calculating probabilities.
    """
    # Validate command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Initialize probability distributions for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Calculate probabilities for all combinations of traits and gene sets
    names = set(people)
    for have_trait in powerset(names):

        # Skip invalid combinations based on provided data
        if any(
            people[person]["trait"] is not None and
            people[person]["trait"] != (person in have_trait)
            for person in names
        ):
            continue

        # Loop through all gene combinations
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Calculate and update probabilities
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Normalize probabilities for output
    normalize(probabilities)

    # Print results in a readable format
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Read gene and trait data from a CSV file into a structured dictionary.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Generate all possible subsets of a given set `s`.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Calculate the joint probability for the provided configuration.
    """
    joint_prob = 1

    # Compute probabilities for each person
    for person in people:
        person_prob = 1
        person_genes = (2 if person in two_genes else
                        1 if person in one_gene else
                        0)
        person_trait = person in have_trait
        mother = people[person]['mother']
        father = people[person]['father']

        # If no parent data is available, use unconditional probabilities
        if not mother and not father:
            person_prob *= PROBS['gene'][person_genes]
        else:
            # Compute inheritance probabilities based on parents
            mother_prob = inherit_prob(mother, one_gene, two_genes)
            father_prob = inherit_prob(father, one_gene, two_genes)

            if person_genes == 2:
                person_prob *= mother_prob * father_prob
            elif person_genes == 1:
                person_prob *= (1 - mother_prob) * father_prob + (1 - father_prob) * mother_prob
            else:
                person_prob *= (1 - mother_prob) * (1 - father_prob)

        # Account for trait probability
        person_prob *= PROBS['trait'][person_genes][person_trait]
        joint_prob *= person_prob

    return joint_prob


def inherit_prob(parent_name, one_gene, two_genes):
    """
    Helper function to compute the probability of passing a mutated gene.
    """
    if parent_name in two_genes:
        return 1 - PROBS['mutation']
    elif parent_name in one_gene:
        return 0.5
    else:
        return PROBS['mutation']


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Update the probability distributions with a new joint probability `p`.
    """
    for person in probabilities:
        person_genes = (2 if person in two_genes else
                        1 if person in one_gene else
                        0)
        person_trait = person in have_trait
        probabilities[person]['gene'][person_genes] += p
        probabilities[person]['trait'][person_trait] += p


def normalize(probabilities):
    """
    Normalize probabilities to ensure they sum to 1.
    """
    for person in probabilities:
        gene_total = sum(probabilities[person]['gene'].values())
        trait_total = sum(probabilities[person]['trait'].values())

        probabilities[person]['gene'] = {gene: prob / gene_total
                                         for gene, prob in probabilities[person]['gene'].items()}
        probabilities[person]['trait'] = {trait: prob / trait_total
                                          for trait, prob in probabilities[person]['trait'].items()}


if __name__ == "__main__":
    main()
