import os
import random
import re
import sys

# Probability that a random surfer clicks a link on the current page
# If not, the surfer jumps to a random page in the corpus.
DAMPING = 0.85
SAMPLES = 10000


def main():
    """Main function to run the PageRank algorithm."""
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")

    corpus = crawl(sys.argv[1])
    
    # Compute PageRank using the sampling method
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

    # Compute PageRank using the iterative method
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and extract links to other pages.

    Returns:
        A dictionary where each key is a page, and the value is a set of all pages
        linked to by that page.
    """
    pages = {}

    # Read all HTML files in the given directory
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            # Extract all href links from the HTML file
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages that exist in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Create a probability distribution over the next page to visit.

    Args:
        corpus: Dictionary mapping pages to their links.
        page: The current page being visited.
        damping_factor: Probability of choosing a linked page.

    Returns:
        A dictionary representing the probability distribution of visiting each page.
    """
    prob_dist = {page_name: 0 for page_name in corpus}

    # If the page has no outgoing links, treat it as linking to all pages
    if not corpus[page]:
        for page_name in prob_dist:
            prob_dist[page_name] = 1 / len(corpus)
        return prob_dist

    # Base probability for randomly selecting any page
    random_prob = (1 - damping_factor) / len(corpus)

    # Additional probability for pages linked from the current page
    link_prob = damping_factor / len(corpus[page])

    for page_name in prob_dist:
        prob_dist[page_name] += random_prob
        if page_name in corpus[page]:
            prob_dist[page_name] += link_prob

    return prob_dist


def sample_pagerank(corpus, damping_factor, n):
    """
    Estimate PageRank values by simulating a random surfer.

    Args:
        corpus: Dictionary mapping pages to their links.
        damping_factor: Probability of following a link.
        n: Number of samples to generate.

    Returns:
        A dictionary with PageRank values for each page.
    """
    # Track the number of visits to each page
    visits = {page_name: 0 for page_name in corpus}

    # Start the surfer on a random page
    curr_page = random.choice(list(visits))
    visits[curr_page] += 1

    # Perform the sampling process
    for _ in range(n - 1):
        trans_model = transition_model(corpus, curr_page, damping_factor)

        # Choose the next page based on the transition probabilities
        rand_val = random.random()
        total_prob = 0

        for page_name, probability in trans_model.items():
            total_prob += probability
            if rand_val <= total_prob:
                curr_page = page_name
                break

        visits[curr_page] += 1

    # Normalize visits to calculate PageRank
    page_ranks = {page_name: visits[page_name] / n for page_name in visits}
    return page_ranks


def iterate_pagerank(corpus, damping_factor):
    """
    Calculate PageRank values iteratively until they converge.

    Args:
        corpus: Dictionary mapping pages to their links.
        damping_factor: Probability of following a link.

    Returns:
        A dictionary with PageRank values for each page.
    """
    num_pages = len(corpus)
    initial_rank = 1 / num_pages
    page_ranks = {page: initial_rank for page in corpus}

    # Handle pages with no links by treating them as linking to all pages
    for page in corpus:
        if not corpus[page]:
            corpus[page] = set(corpus.keys())

    # Repeat until ranks converge
    converged = False
    while not converged:
        new_ranks = {}
        for page in corpus:
            # Base rank from random selection
            rank = (1 - damping_factor) / num_pages
            # Add contributions from linked pages
            for linking_page in corpus:
                if page in corpus[linking_page]:
                    rank += damping_factor * (page_ranks[linking_page] / len(corpus[linking_page]))
            new_ranks[page] = rank

        # Check if ranks have converged
        converged = all(
            abs(new_ranks[page] - page_ranks[page]) < 0.001
            for page in page_ranks
        )
        page_ranks = new_ranks

    return page_ranks


if __name__ == "__main__":
    main()
