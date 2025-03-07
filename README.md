# Bee AI Fact Consolidator

A command-line tool that uses k-means clustering and LLM-based consolidation to deduplicate and consolidate personal facts from Bee AI.

## Features

- Uses k-means clustering to group similar facts
- Sends grouped facts to a locally hosted LLM for consolidation
- Interactive approval of consolidation results
- Batch processing via command line
- Utilizes the Bee AI Facts API to manage facts
- Interactive credential prompting for easy setup

## Requirements

- Python 3.9+
- Access to Bee AI API
- Local LLM running via LM Studio (or compatible OpenAI API interface). I used LM Studio with Qwen 2.5 7B Instruct.

## Installation

### Using Homebrew (macOS/Linux)

```bash
# Add the tap (only needed once)
brew tap imatson9119/bee-ai-fact-consolidator https://github.com/imatson9119/bee-ai-fact-consolidator

# Install the package
brew install bee-ai-fact-consolidator
```

### Manual Installation

1. Clone this repository:
   ```
   git clone https://github.com/imatson9119/bee-ai-fact-consolidator.git
   cd bee-ai-fact-consolidator
   ```

2. Set up the virtual environment using Pipenv:
   ```
   pipenv install
   ```
   
   Or install using pip:
   ```
   pip install -e .
   ```

3. Create a `.env` file based on the `.env.example`:
   ```
   cp .env.example .env
   ```

4. Edit the `.env` file and add your Bee AI API token.

## Usage

1. Make sure your local LLM is running with LM Studio at `http://localhost:1234/v1`

2. Run the consolidator:
   ```
   # If installed with Homebrew or pip
   bee-fact-consolidator
   
   # If using Pipenv
   pipenv run python fact_consolidator.py
   ```

3. If this is your first time running the tool, you will be prompted for:
   - Your Bee AI API token (get one from https://developer.bee.computer)
   - Your LLM API URL (default: http://localhost:1234/v1)

### Command-line options

- `--min-cluster-size`: Minimum number of facts in a cluster to consider for consolidation (default: 2)
- `--confirmed-only`: Only process confirmed facts
- `--auto-approve`: Automatically approve all consolidations without prompting
- `--dry-run`: Don't make any changes, just show what would be done
- `--similarity-threshold`: Similarity threshold for clustering (0.0-1.0) (default: 0.5)
- `--use-kmeans`: Use k-means instead of agglomerative clustering (default is agglomerative)
- `--debug`: Enable debug logging for troubleshooting

### Examples

Process all facts with default settings (interactive approval):
```
bee-fact-consolidator
```

Process only confirmed facts:
```
bee-fact-consolidator --confirmed-only
```

Automatically approve all consolidations:
```
bee-fact-consolidator --auto-approve
```

Dry run to see what would be consolidated without making changes:
```
bee-fact-consolidator --dry-run
```

Adjust the similarity threshold for clustering:
```
bee-fact-consolidator --similarity-threshold 0.7
```

Debug mode for troubleshooting:
```
bee-fact-consolidator --debug
```

## How It Works

1. **Fact Retrieval**: Facts are retrieved from the Bee AI API.
2. **Clustering**: K-means clustering with TF-IDF vectorization groups similar facts.
3. **Consolidation**: For each cluster with at least `min-cluster-size` facts, the local LLM generates consolidated facts.
4. **User Approval**: The tool displays original and consolidated facts for user approval.
5. **Update**: If approved, original facts are deleted and new consolidated facts are created.

## Example

Given these original facts:
- Ian is friends with someone named Scott
- Ian has a friend named Scott
- Scott is friends with Ian
- Ian's friend Scott is the best man at his wedding

The tool might suggest consolidation to:
- Ian is close friends with someone named Scott
- Ian's friend Scott is the best man at his wedding 

## Troubleshooting

### API Format Issues
If you encounter errors related to the API response format, enable debug mode with `--debug` to see detailed error messages and the actual response structure.

### LLM Connection
Make sure the LLM is properly running with LM Studio at the URL specified in your `.env` file (default: http://localhost:1234/v1). You can test your LLM connection by running:
```
./test_llm_integration.py
```

### Testing Without API Access
If you don't have API access yet or want to test locally:
```
./test_consolidation.py
```

This runs the clustering and consolidation with mock data without making actual API calls. 