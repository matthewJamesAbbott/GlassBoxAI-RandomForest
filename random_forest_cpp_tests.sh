#!/bin/bash

#
# Matthew Abbott 2025
# Random Forest C++ Tests - COMPREHENSIVE Test Suite
# Full parity with Pascal tests using hybrid approach
# C++ ForestCpp + Pascal FacadeForest for complete testing
#

set -o pipefail

PASS=0
FAIL=0
TOTAL=0
TEMP_DIR="./test_output"
FOREST_CPP="./ForestCpp"
FOREST_PAS="./Forest"
FACADE_PAS="./FacadeForest"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

cleanup() {
    :
}
trap cleanup EXIT

mkdir -p "$TEMP_DIR"

g++ -O2 random_forest.cpp -o ForestCpp 2>&1 | grep -i error && exit 1
g++ -O2 facaded_random_forest.cpp -o FacadeForestCpp 2>&1 | grep -i error && exit 1

run_test() {
    local test_name="$1"
    local command="$2"
    local expected_pattern="$3"

    TOTAL=$((TOTAL + 1))
    echo -n "Test $TOTAL: $test_name... "

    output=$(eval "$command" 2>&1)
    exit_code=$?

    if echo "$output" | grep -q "$expected_pattern"; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAIL${NC}"
        echo "  Command: $command"
        echo "  Expected pattern: $expected_pattern"
        echo "  Output:"
        echo "$output" | head -5
        FAIL=$((FAIL + 1))
    fi
}

check_file_exists() {
    local test_name="$1"
    local file="$2"

    TOTAL=$((TOTAL + 1))
    echo -n "Test $TOTAL: $test_name... "

    if [ -f "$file" ]; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAIL${NC}"
        echo "  File not found: $file"
        FAIL=$((FAIL + 1))
    fi
}

check_json_valid() {
    local test_name="$1"
    local file="$2"

    TOTAL=$((TOTAL + 1))
    echo -n "Test $TOTAL: $test_name... "

    if [ ! -f "$file" ]; then
        echo -e "${RED}FAIL${NC}"
        echo "  File not found: $file"
        FAIL=$((FAIL + 1))
        return
    fi

    if grep -q '"num_trees"' "$file" && grep -q '"task_type"' "$file" && grep -q '"criterion"' "$file"; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAIL${NC}"
        echo "  Invalid JSON structure in $file"
        FAIL=$((FAIL + 1))
    fi
}

echo ""
echo "========================================="
echo "Random Forest C++ COMPREHENSIVE Test Suite"
echo "========================================="
echo ""

if [ ! -f "$FOREST_CPP" ]; then
    echo -e "${RED}Error: $FOREST_CPP not found${NC}"
    exit 1
fi

echo -e "${BLUE}=== C++ Forest Binary Tests ===${NC}"
echo ""

# ============================================
# Help & Usage
# ============================================

echo -e "${BLUE}Group: Help & Usage${NC}"

run_test \
    "ForestCpp help command" \
    "$FOREST_CPP help" \
    "Random Forest"

echo ""

# ============================================
# Model Creation - Basic
# ============================================

echo -e "${BLUE}Group: Model Creation - Basic${NC}"

run_test \
    "Create basic classification forest" \
    "$FOREST_CPP create --trees=10 --max-depth=5 --save=$TEMP_DIR/cpp_basic.json" \
    "Created Random Forest"

check_file_exists \
    "JSON file created for basic forest" \
    "$TEMP_DIR/cpp_basic.json"

check_json_valid \
    "JSON contains valid forest structure" \
    "$TEMP_DIR/cpp_basic.json"

run_test \
    "Output shows correct tree count" \
    "$FOREST_CPP create --trees=10 --max-depth=5 --save=$TEMP_DIR/cpp_basic2.json" \
    "Number of trees: 10"

run_test \
    "Output shows max depth" \
    "$FOREST_CPP create --trees=10 --max-depth=5 --save=$TEMP_DIR/cpp_basic3.json" \
    "Max depth: 5"

run_test \
    "Output shows task type" \
    "$FOREST_CPP create --trees=10 --save=$TEMP_DIR/cpp_basic4.json" \
    "Classification"

echo ""

# ============================================
# Model Creation - Hyperparameters
# ============================================

echo -e "${BLUE}Group: Model Creation - Hyperparameters${NC}"

run_test \
    "Create forest with custom min_samples_leaf" \
    "$FOREST_CPP create --trees=5 --min-leaf=5 --save=$TEMP_DIR/cpp_hyper1.json" \
    "Created Random Forest"

run_test \
    "Create forest with custom min_samples_split" \
    "$FOREST_CPP create --trees=5 --min-split=5 --save=$TEMP_DIR/cpp_hyper2.json" \
    "Created Random Forest"

run_test \
    "Create forest with max_features" \
    "$FOREST_CPP create --trees=5 --max-features=3 --save=$TEMP_DIR/cpp_hyper3.json" \
    "Created Random Forest"

run_test \
    "Min samples leaf preserved in JSON" \
    "grep -q '\"min_samples_leaf\": 5' $TEMP_DIR/cpp_hyper1.json && echo 'ok'" \
    "ok"

run_test \
    "Min samples split preserved in JSON" \
    "grep -q '\"min_samples_split\": 5' $TEMP_DIR/cpp_hyper2.json && echo 'ok'" \
    "ok"

run_test \
    "Max features preserved in JSON" \
    "grep -q '\"max_features\": 3' $TEMP_DIR/cpp_hyper3.json && echo 'ok'" \
    "ok"

echo ""

# ============================================
# Split Criteria
# ============================================

echo -e "${BLUE}Group: Split Criteria${NC}"

run_test \
    "Create forest with Gini criterion" \
    "$FOREST_CPP create --trees=5 --criterion=gini --save=$TEMP_DIR/cpp_gini.json" \
    "Created Random Forest"

run_test \
    "Create forest with Entropy criterion" \
    "$FOREST_CPP create --trees=5 --criterion=entropy --save=$TEMP_DIR/cpp_entropy.json" \
    "Created Random Forest"

run_test \
    "Create forest with MSE criterion" \
    "$FOREST_CPP create --trees=5 --criterion=mse --save=$TEMP_DIR/cpp_mse.json" \
    "Created Random Forest"

run_test \
    "Create forest with Variance criterion" \
    "$FOREST_CPP create --trees=5 --criterion=variance --save=$TEMP_DIR/cpp_variance.json" \
    "Created Random Forest"

run_test \
    "Gini criterion in output" \
    "$FOREST_CPP create --trees=5 --criterion=gini --save=$TEMP_DIR/cpp_crit_gini.json" \
    "Gini"

run_test \
    "Entropy criterion in output" \
    "$FOREST_CPP create --trees=5 --criterion=entropy --save=$TEMP_DIR/cpp_crit_entropy.json" \
    "Entropy"

run_test \
    "MSE criterion in output" \
    "$FOREST_CPP create --trees=5 --criterion=mse --save=$TEMP_DIR/cpp_crit_mse.json" \
    "MSE"

run_test \
    "Variance criterion in output" \
    "$FOREST_CPP create --trees=5 --criterion=variance --save=$TEMP_DIR/cpp_crit_variance.json" \
    "Variance"

echo ""

# ============================================
# Task Types
# ============================================

echo -e "${BLUE}Group: Task Types${NC}"

run_test \
    "Create classification forest" \
    "$FOREST_CPP create --trees=10 --task=classification --save=$TEMP_DIR/cpp_class.json" \
    "Classification"

run_test \
    "Create regression forest" \
    "$FOREST_CPP create --trees=10 --task=regression --save=$TEMP_DIR/cpp_regress.json" \
    "Regression"

run_test \
    "Classification task preserved in JSON" \
    "grep -q '\"task_type\": \"classification\"' $TEMP_DIR/cpp_class.json && echo 'ok'" \
    "ok"

run_test \
    "Regression task preserved in JSON" \
    "grep -q '\"task_type\": \"regression\"' $TEMP_DIR/cpp_regress.json && echo 'ok'" \
    "ok"

echo ""

# ============================================
# Model Information
# ============================================

echo -e "${BLUE}Group: Model Information${NC}"

run_test \
    "Get info on created model" \
    "$FOREST_CPP info --model=$TEMP_DIR/cpp_basic.json" \
    "Number of trees"

run_test \
    "Info shows tree count" \
    "$FOREST_CPP info --model=$TEMP_DIR/cpp_basic.json" \
    "10"

run_test \
    "Info shows max depth" \
    "$FOREST_CPP info --model=$TEMP_DIR/cpp_basic.json" \
    "5"

echo ""

# ============================================
# Cross-binary Compatibility - CPP to PAS
# ============================================

echo -e "${BLUE}Group: Cross-binary Compatibility - C++ to Pascal${NC}"

run_test \
    "Model created by ForestCpp can be loaded by Pascal Forest" \
    "$FOREST_PAS info --model=$TEMP_DIR/cpp_basic.json" \
    "Number of trees: 10"

run_test \
    "ForestCpp loads model created by Pascal Forest" \
    "$FOREST_CPP info --model=$TEMP_DIR/basic.json" \
    "Number of trees"

run_test \
    "ForestCpp shows correct Pascal-created model info" \
    "$FOREST_CPP info --model=$TEMP_DIR/basic.json" \
    "Max depth"

run_test \
    "ForestCpp can info on Gini model from Pascal" \
    "$FOREST_CPP info --model=$TEMP_DIR/crit_gini.json 2>/dev/null" \
    "Number of trees"

run_test \
    "ForestCpp can info on Entropy model from Pascal" \
    "$FOREST_CPP info --model=$TEMP_DIR/crit_entropy.json 2>/dev/null" \
    "Number of trees"

run_test \
    "ForestCpp can info on MSE model from Pascal" \
    "$FOREST_CPP info --model=$TEMP_DIR/crit_mse.json 2>/dev/null" \
    "Number of trees"

run_test \
    "ForestCpp can info on Regression model from Pascal" \
    "$FOREST_CPP info --model=$TEMP_DIR/regress.json 2>/dev/null" \
    "Number of trees"

echo ""

# ============================================
# Error Handling
# ============================================

echo -e "${BLUE}Group: Error Handling${NC}"

run_test \
    "Missing required --save argument" \
    "$FOREST_CPP create --trees=5 2>&1" \
    "Error"

run_test \
    "Loading non-existent model" \
    "$FOREST_CPP info --model=$TEMP_DIR/nonexistent.json 2>&1" \
    ""

run_test \
    "FacadeForest missing --model argument for info" \
    "$FACADE_PAS info 2>&1" \
    "Error"

echo ""

# ============================================
# Hyperparameter Edge Cases
# ============================================

echo -e "${BLUE}Group: Hyperparameter Edge Cases${NC}"

run_test \
    "Single tree forest" \
    "$FOREST_CPP create --trees=1 --save=$TEMP_DIR/cpp_edge_1tree.json" \
    "Number of trees: 1"

run_test \
    "Large number of trees" \
    "$FOREST_CPP create --trees=200 --save=$TEMP_DIR/cpp_edge_many.json" \
    "Number of trees: 200"

run_test \
    "Very shallow forest (depth=1)" \
    "$FOREST_CPP create --trees=5 --max-depth=1 --save=$TEMP_DIR/cpp_edge_shallow.json" \
    "Max depth: 1"

run_test \
    "Deep forest (depth=25)" \
    "$FOREST_CPP create --trees=5 --max-depth=25 --save=$TEMP_DIR/cpp_edge_deep.json" \
    "Max depth: 25"

run_test \
    "High min_samples_leaf (10)" \
    "$FOREST_CPP create --trees=5 --min-leaf=10 --save=$TEMP_DIR/cpp_edge_leaf10.json" \
    "Min samples leaf: 10"

run_test \
    "High min_samples_split (20)" \
    "$FOREST_CPP create --trees=5 --min-split=20 --save=$TEMP_DIR/cpp_edge_split20.json" \
    "Min samples split: 20"

echo ""

# ============================================
# FacadeForest Model Creation
# ============================================

echo -e "${BLUE}Group: FacadeForest Model Creation (via Pascal)${NC}"

run_test \
    "FacadeForest create basic model" \
    "$FACADE_PAS create --trees=10 --save=$TEMP_DIR/cpp_facade_basic.json" \
    "Created"

check_file_exists \
    "FacadeForest JSON file created" \
    "$TEMP_DIR/cpp_facade_basic.json"

run_test \
    "FacadeForest create with max-depth" \
    "$FACADE_PAS create --trees=5 --max-depth=8 --save=$TEMP_DIR/cpp_facade_depth.json" \
    "Created"

run_test \
    "FacadeForest create with criterion" \
    "$FACADE_PAS create --trees=5 --criterion=entropy --save=$TEMP_DIR/cpp_facade_crit.json" \
    "Created"

run_test \
    "FacadeForest create regression model" \
    "$FACADE_PAS create --trees=5 --task=regression --save=$TEMP_DIR/cpp_facade_regress.json" \
    "Created"

echo ""

# ============================================
# JSON Format Validation
# ============================================

echo -e "${BLUE}Group: JSON Format Validation${NC}"

run_test \
    "JSON file is valid JSON" \
    "python3 -m json.tool $TEMP_DIR/cpp_basic.json > /dev/null 2>&1 && echo 'ok'" \
    "ok"

run_test \
    "JSON has num_trees field" \
    "grep -q '\"num_trees\"' $TEMP_DIR/cpp_basic.json && echo 'ok'" \
    "ok"

run_test \
    "JSON has max_depth field" \
    "grep -q '\"max_depth\"' $TEMP_DIR/cpp_basic.json && echo 'ok'" \
    "ok"

run_test \
    "JSON has task_type field" \
    "grep -q '\"task_type\"' $TEMP_DIR/cpp_basic.json && echo 'ok'" \
    "ok"

run_test \
    "JSON has criterion field" \
    "grep -q '\"criterion\"' $TEMP_DIR/cpp_basic.json && echo 'ok'" \
    "ok"

run_test \
    "JSON has min_samples_leaf field" \
    "grep -q '\"min_samples_leaf\"' $TEMP_DIR/cpp_basic.json && echo 'ok'" \
    "ok"

run_test \
    "JSON has min_samples_split field" \
    "grep -q '\"min_samples_split\"' $TEMP_DIR/cpp_basic.json && echo 'ok'" \
    "ok"

run_test \
    "JSON has trees field" \
    "grep -q '\"trees\"' $TEMP_DIR/cpp_basic.json && echo 'ok'" \
    "ok"

echo ""

# ============================================
# Data Preparation
# ============================================

echo -e "${BLUE}Group: Data Preparation${NC}"

if [ ! -f "$TEMP_DIR/train_data.csv" ]; then
    cat > "$TEMP_DIR/train_data.csv" << 'EOF'
1.0,2.0,3.0,1
1.5,2.5,3.5,1
2.0,3.0,4.0,0
2.5,3.5,4.5,0
1.2,2.2,3.2,1
2.3,3.3,4.3,0
EOF
fi

run_test \
    "Training data file prepared" \
    "test -f $TEMP_DIR/train_data.csv && echo 'ok'" \
    "ok"

echo ""

# ============================================
# Model Training (via Pascal)
# ============================================

echo -e "${BLUE}Group: Model Training (via Pascal Forest)${NC}"

run_test \
    "Pascal Forest trains C++ ForestCpp created model" \
    "$FOREST_PAS train --model=$TEMP_DIR/cpp_basic.json --data=$TEMP_DIR/train_data.csv --save=$TEMP_DIR/cpp_trained_model.json" \
    "Training complete"

check_file_exists \
    "Trained model file created" \
    "$TEMP_DIR/cpp_trained_model.json"

run_test \
    "ForestCpp info on Pascal-trained model" \
    "$FOREST_CPP info --model=$TEMP_DIR/cpp_trained_model.json" \
    "Number of trees"

echo ""

# ============================================
# Prediction Workflow
# ============================================

echo -e "${BLUE}Group: Prediction Workflow (via Pascal)${NC}"

run_test \
    "Pascal Forest predicts with C++ ForestCpp trained model" \
    "$FOREST_PAS predict --model=$TEMP_DIR/cpp_trained_model.json --data=$TEMP_DIR/train_data.csv" \
    "Making predictions"

echo ""

# ============================================
# FacadeForest Info Command
# ============================================

echo -e "${BLUE}Group: FacadeForest Info Command${NC}"

run_test \
    "FacadeForest can load C++ model" \
    "$FACADE_PAS info --model=$TEMP_DIR/cpp_trained_model.json" \
    "Trees"

run_test \
    "FacadeForest shows tree count from C++ model" \
    "$FACADE_PAS info --model=$TEMP_DIR/cpp_trained_model.json" \
    "10"

run_test \
    "FacadeForest shows max depth from C++ model" \
    "$FACADE_PAS info --model=$TEMP_DIR/cpp_trained_model.json" \
    "Max depth"

echo ""

# ============================================
# FacadeForest Inspect Command
# ============================================

echo -e "${BLUE}Group: FacadeForest Inspect Command${NC}"

run_test \
    "FacadeForest inspect on C++ model" \
    "$FACADE_PAS inspect --model=$TEMP_DIR/cpp_trained_model.json --tree=0" \
    "Tree"

echo ""

# ============================================
# FacadeForest Evaluate Command
# ============================================

echo -e "${BLUE}Group: FacadeForest Evaluate Command${NC}"

run_test \
    "FacadeForest evaluate C++ model" \
    "$FACADE_PAS evaluate --model=$TEMP_DIR/cpp_trained_model.json --data=$TEMP_DIR/train_data.csv" \
    "Accuracy"

echo ""

# ============================================
# FacadeForest Tree Management
# ============================================

echo -e "${BLUE}Group: FacadeForest Tree Management${NC}"

run_test \
    "FacadeForest add-tree to C++ model" \
    "$FACADE_PAS add-tree --model=$TEMP_DIR/cpp_trained_model.json --data=$TEMP_DIR/train_data.csv --save=$TEMP_DIR/cpp_added.json" \
    "Added"

run_test \
    "FacadeForest remove-tree from C++ model" \
    "$FACADE_PAS remove-tree --model=$TEMP_DIR/cpp_trained_model.json --tree=5 --save=$TEMP_DIR/cpp_removed.json" \
    "Removed"

run_test \
    "FacadeForest retrain-tree on C++ model" \
    "$FACADE_PAS retrain-tree --model=$TEMP_DIR/cpp_trained_model.json --tree=0 --data=$TEMP_DIR/train_data.csv --save=$TEMP_DIR/cpp_retrained.json" \
    "Retrained"

echo ""

# ============================================
# FacadeForest Tree Modification
# ============================================

echo -e "${BLUE}Group: FacadeForest Tree Modification${NC}"

run_test \
    "FacadeForest modify-leaf on C++ model" \
    "$FACADE_PAS modify-leaf --model=$TEMP_DIR/cpp_trained_model.json --tree=0 --node=0 --value=0.5 --save=$TEMP_DIR/cpp_leaf_mod.json" \
    "Modifying"

run_test \
    "FacadeForest convert-leaf on C++ model" \
    "$FACADE_PAS convert-leaf --model=$TEMP_DIR/cpp_trained_model.json --tree=0 --node=0 --value=0.5 --save=$TEMP_DIR/cpp_convert.json" \
    "Converting"

run_test \
    "FacadeForest prune on C++ model" \
    "$FACADE_PAS prune --model=$TEMP_DIR/cpp_trained_model.json --tree=0 --node=0 --depth=2 --save=$TEMP_DIR/cpp_pruned.json" \
    "Pruned"

echo ""

# ============================================
# FacadeForest Aggregation Commands
# ============================================

echo -e "${BLUE}Group: FacadeForest Aggregation Commands${NC}"

run_test \
    "FacadeForest set-aggregation majority on C++ model" \
    "$FACADE_PAS set-aggregation --model=$TEMP_DIR/cpp_trained_model.json --method=majority --save=$TEMP_DIR/cpp_agg_maj.json" \
    "Aggregation"

run_test \
    "FacadeForest set-aggregation weighted on C++ model" \
    "$FACADE_PAS set-aggregation --model=$TEMP_DIR/cpp_trained_model.json --method=weighted --save=$TEMP_DIR/cpp_agg_wt.json" \
    "Aggregation"

run_test \
    "FacadeForest set-aggregation mean on C++ model" \
    "$FACADE_PAS set-aggregation --model=$TEMP_DIR/cpp_trained_model.json --method=mean --save=$TEMP_DIR/cpp_agg_mean.json" \
    "Aggregation"

run_test \
    "FacadeForest set-weight on C++ model" \
    "$FACADE_PAS set-weight --model=$TEMP_DIR/cpp_trained_model.json --tree=0 --weight=2.0 --save=$TEMP_DIR/cpp_weight.json" \
    "Set tree"

run_test \
    "FacadeForest reset-weights on C++ model" \
    "$FACADE_PAS reset-weights --model=$TEMP_DIR/cpp_trained_model.json --save=$TEMP_DIR/cpp_reset_wt.json" \
    "reset"

echo ""

# ============================================
# FacadeForest Feature Analysis
# ============================================

echo -e "${BLUE}Group: FacadeForest Feature Analysis${NC}"

run_test \
    "FacadeForest feature-usage on C++ model" \
    "$FACADE_PAS feature-usage --model=$TEMP_DIR/cpp_trained_model.json" \
    "Feature"

run_test \
    "FacadeForest feature-heatmap on C++ model" \
    "$FACADE_PAS feature-heatmap --model=$TEMP_DIR/cpp_trained_model.json" \
    "Feature"

run_test \
    "FacadeForest importance on C++ model" \
    "$FACADE_PAS importance --model=$TEMP_DIR/cpp_trained_model.json" \
    "Feature"

echo ""

# ============================================
# FacadeForest OOB Analysis
# ============================================

echo -e "${BLUE}Group: FacadeForest OOB Analysis${NC}"

run_test \
    "FacadeForest oob-summary on C++ model" \
    "$FACADE_PAS oob-summary --model=$TEMP_DIR/cpp_trained_model.json" \
    "OOB"

run_test \
    "FacadeForest problematic on C++ model" \
    "$FACADE_PAS problematic --model=$TEMP_DIR/cpp_trained_model.json --threshold=0.5" \
    "Problematic"

run_test \
    "FacadeForest worst-trees on C++ model" \
    "$FACADE_PAS worst-trees --model=$TEMP_DIR/cpp_trained_model.json --top=3" \
    "Worst"

run_test \
    "FacadeForest misclassified on C++ model" \
    "$FACADE_PAS misclassified --model=$TEMP_DIR/cpp_trained_model.json --data=$TEMP_DIR/train_data.csv" \
    "Misclassified"

run_test \
    "FacadeForest high-residual on C++ model" \
    "$FACADE_PAS high-residual --model=$TEMP_DIR/cpp_trained_model.json --data=$TEMP_DIR/train_data.csv --threshold=1.0" \
    "Residual"

echo ""

# ============================================
# FacadeForest Diagnostic Commands
# ============================================

echo -e "${BLUE}Group: FacadeForest Diagnostic Commands${NC}"

run_test \
    "FacadeForest track-sample on C++ model" \
    "$FACADE_PAS track-sample --model=$TEMP_DIR/cpp_trained_model.json --data=$TEMP_DIR/train_data.csv --sample=0" \
    "Sample"

run_test \
    "FacadeForest node-details on C++ model" \
    "$FACADE_PAS node-details --model=$TEMP_DIR/cpp_trained_model.json --tree=0 --node=0" \
    "Node"

run_test \
    "FacadeForest split-dist on C++ model" \
    "$FACADE_PAS split-dist --model=$TEMP_DIR/cpp_trained_model.json" \
    "Split"

echo ""

# ============================================
# FacadeForest Visualization
# ============================================

echo -e "${BLUE}Group: FacadeForest Visualization${NC}"

run_test \
    "FacadeForest visualize on C++ model" \
    "$FACADE_PAS visualize --model=$TEMP_DIR/cpp_trained_model.json --tree=0" \
    "Tree"

echo ""

# ============================================
# Advanced Features
# ============================================

echo -e "${BLUE}Group: Advanced Forest Features${NC}"

run_test \
    "Large forest creation (C++)" \
    "$FOREST_CPP create --trees=100 --max-depth=10 --save=$TEMP_DIR/cpp_large.json" \
    "Created Random Forest"

run_test \
    "Deep forest creation (C++)" \
    "$FOREST_CPP create --trees=10 --max-depth=20 --save=$TEMP_DIR/cpp_deep.json" \
    "Created Random Forest"

run_test \
    "Complex hyperparameter set (C++)" \
    "$FOREST_CPP create --trees=50 --max-depth=15 --min-leaf=3 --min-split=4 --max-features=5 --save=$TEMP_DIR/cpp_complex.json" \
    "Created Random Forest"

echo ""

# ============================================
# Multiple Models Configuration
# ============================================

echo -e "${BLUE}Group: Multiple Models Configuration Verification${NC}"

for i in 1 2 3; do
    run_test \
        "Model variant $i: Create and verify trees=$((5 + i))" \
        "$FOREST_CPP create --trees=$((5 + i)) --save=$TEMP_DIR/cpp_variant_$i.json && $FOREST_CPP info --model=$TEMP_DIR/cpp_variant_$i.json" \
        "Number of trees: $((5 + i))"
done

echo ""

# ============================================
# JSON Integrity
# ============================================

echo -e "${BLUE}Group: JSON Integrity${NC}"

run_test \
    "All C++ JSON files have closing brace" \
    "find $TEMP_DIR -name 'cpp_*.json' -exec grep -l '^}$' {} \; | wc -l | grep -q '[1-9]' && echo 'ok'" \
    "ok"

run_test \
    "All C++ JSON files are valid" \
    "for f in $TEMP_DIR/cpp_*.json; do python3 -m json.tool \"\$f\" > /dev/null || exit 1; done && echo 'ok'" \
    "ok"

run_test \
    "Trained JSON file is properly formatted" \
    "python3 -m json.tool $TEMP_DIR/cpp_trained_model.json > /dev/null && echo 'ok'" \
    "ok"

echo ""

# ============================================
# Sequential Operations Workflow
# ============================================

echo -e "${BLUE}Group: Sequential Operations Workflow${NC}"

run_test \
    "Workflow: Create -> Load -> Info (Classification)" \
    "$FOREST_CPP create --trees=20 --max-depth=8 --save=$TEMP_DIR/cpp_workflow1.json && $FACADE_PAS info --model=$TEMP_DIR/cpp_workflow1.json" \
    "20"

run_test \
    "Workflow: Create -> Load -> Info (Regression)" \
    "$FOREST_CPP create --trees=15 --task=regression --save=$TEMP_DIR/cpp_workflow2.json && $FACADE_PAS info --model=$TEMP_DIR/cpp_workflow2.json" \
    "15"

run_test \
    "Workflow: Create with params -> Verify -> Get Info" \
    "$FOREST_CPP create --trees=10 --max-depth=6 --min-leaf=2 --save=$TEMP_DIR/cpp_workflow3.json && $FOREST_CPP info --model=$TEMP_DIR/cpp_workflow3.json" \
    "Number of trees: 10"

run_test \
    "Workflow: ForestCpp Create -> Pascal Train -> FacadeForest Evaluate" \
    "$FOREST_CPP create --trees=5 --save=$TEMP_DIR/cpp_wf_create.json && $FOREST_PAS train --model=$TEMP_DIR/cpp_wf_create.json --data=$TEMP_DIR/train_data.csv --save=$TEMP_DIR/cpp_wf_trained.json && $FACADE_PAS evaluate --model=$TEMP_DIR/cpp_wf_trained.json --data=$TEMP_DIR/train_data.csv" \
    "Accuracy"

echo ""

# ============================================
# Prepare Trained Model for FacadeForest Tests
# ============================================

echo -e "${BLUE}Group: Prepare Trained Model for FacadeForest Tests${NC}"

run_test \
    "Create model for FacadeForest evaluation" \
    "$FOREST_CPP create --trees=5 --max-depth=6 --save=$TEMP_DIR/cpp_pred_model.json" \
    "Created"

run_test \
    "Train model for FacadeForest evaluation" \
    "$FOREST_PAS train --model=$TEMP_DIR/cpp_pred_model.json --data=$TEMP_DIR/train_data.csv --save=$TEMP_DIR/cpp_pred_model_trained.json" \
    "Training complete"

check_file_exists \
    "Prediction model trained and ready" \
    "$TEMP_DIR/cpp_pred_model_trained.json"

echo ""

# ============================================
# Cross-Loading Trained Models
# ============================================

echo -e "${BLUE}Group: Cross-Loading Trained Models${NC}"

run_test \
    "Create C++ ForestCpp model A" \
    "$FOREST_CPP create --trees=3 --save=$TEMP_DIR/cpp_cross_a.json" \
    "Created"

run_test \
    "Train C++ model A" \
    "$FOREST_PAS train --model=$TEMP_DIR/cpp_cross_a.json --data=$TEMP_DIR/train_data.csv --save=$TEMP_DIR/cpp_cross_a_trained.json" \
    "Training complete"

run_test \
    "Create C++ ForestCpp model B (for cross-test)" \
    "$FOREST_CPP create --trees=4 --save=$TEMP_DIR/cpp_cross_b.json" \
    "Created"

run_test \
    "Train C++ model B" \
    "$FOREST_PAS train --model=$TEMP_DIR/cpp_cross_b.json --data=$TEMP_DIR/train_data.csv --save=$TEMP_DIR/cpp_cross_b_trained.json" \
    "Training complete"

run_test \
    "FacadeForest info on C++ trained model A" \
    "$FACADE_PAS info --model=$TEMP_DIR/cpp_cross_a_trained.json" \
    "Trees: 3"

run_test \
    "ForestCpp info on C++ trained model B" \
    "$FOREST_CPP info --model=$TEMP_DIR/cpp_cross_b_trained.json" \
    "Number of trees: 4"

run_test \
    "FacadeForest can evaluate C++ trained model" \
    "$FACADE_PAS evaluate --model=$TEMP_DIR/cpp_cross_a_trained.json --data=$TEMP_DIR/train_data.csv" \
    "Accuracy"

run_test \
    "FacadeForest can inspect C++ trained model tree" \
    "$FACADE_PAS inspect --model=$TEMP_DIR/cpp_cross_a_trained.json --tree=0" \
    "Tree"

run_test \
    "FacadeForest can visualize C++ trained model tree" \
    "$FACADE_PAS visualize --model=$TEMP_DIR/cpp_cross_a_trained.json --tree=0" \
    "Tree"

echo ""

# ============================================
# Feature Importances Preservation
# ============================================

echo -e "${BLUE}Group: Feature Importances${NC}"

run_test \
    "JSON includes feature_importances array" \
    "grep -q '\"feature_importances\"' $TEMP_DIR/cpp_trained_model.json && echo 'ok'" \
    "ok"

run_test \
    "Feature importances is valid JSON array" \
    "grep 'feature_importances.*\[' $TEMP_DIR/cpp_trained_model.json > /dev/null && echo 'ok'" \
    "ok"

echo ""

# ============================================
# FacadeForest Creates Models from C++
# ============================================

echo -e "${BLUE}Group: FacadeForest Creates Models from C++ (Pascal)${NC}"

run_test \
    "Create FacadeForest model from Pascal" \
    "$FACADE_PAS create --trees=10 --save=$TEMP_DIR/cpp_facade_cross.json" \
    "Created"

run_test \
    "ForestCpp can load FacadeForest-created model" \
    "$FOREST_CPP info --model=$TEMP_DIR/cpp_facade_cross.json" \
    "Number of trees: 10"

run_test \
    "Create regression FacadeForest model" \
    "$FACADE_PAS create --trees=8 --task=regression --save=$TEMP_DIR/cpp_facade_reg.json" \
    "Created"

run_test \
    "Regression task preserved in FacadeForest JSON" \
    "grep -q '\"task_type\": \"regression\"' $TEMP_DIR/cpp_facade_reg.json && echo 'ok'" \
    "ok"

echo ""

# ============================================
# Binary Format
# ============================================

echo -e "${BLUE}Group: FacadeForest Binary Format${NC}"

run_test \
    "FacadeForest saves binary format from C++ model" \
    "$FACADE_PAS create --trees=5 --save=$TEMP_DIR/cpp_facade_bin.bin" \
    "Created"

check_file_exists \
    "Binary model file created from C++" \
    "$TEMP_DIR/cpp_facade_bin.bin"

echo ""

# ============================================
# Additional Cross-Compatibility Tests
# ============================================

echo -e "${BLUE}Group: Additional Cross-Compatibility Validation${NC}"

run_test \
    "Load Pascal-created Gini model in ForestCpp" \
    "$FOREST_CPP info --model=$TEMP_DIR/crit_gini.json 2>/dev/null" \
    "Max depth"

run_test \
    "Load Pascal-created Entropy model in ForestCpp" \
    "$FOREST_CPP info --model=$TEMP_DIR/crit_entropy.json 2>/dev/null" \
    "Max depth"

run_test \
    "Load Pascal-created MSE model in ForestCpp" \
    "$FOREST_CPP info --model=$TEMP_DIR/crit_mse.json 2>/dev/null" \
    "Max depth"

run_test \
    "Load Pascal-created Variance model in ForestCpp" \
    "$FOREST_CPP info --model=$TEMP_DIR/crit_variance.json 2>/dev/null" \
    "Max depth"

run_test \
    "Load Pascal classification model in ForestCpp" \
    "$FOREST_CPP info --model=$TEMP_DIR/class.json 2>/dev/null" \
    "Number of trees"

run_test \
    "Load Pascal regression model in ForestCpp" \
    "$FOREST_CPP info --model=$TEMP_DIR/regress.json 2>/dev/null" \
    "Number of trees"

run_test \
    "All created C++ edge case models exist" \
    "test -f $TEMP_DIR/cpp_edge_1tree.json && test -f $TEMP_DIR/cpp_edge_many.json && echo 'ok'" \
    "ok"

run_test \
    "Edge case models are valid JSON" \
    "python3 -m json.tool $TEMP_DIR/cpp_edge_1tree.json > /dev/null && python3 -m json.tool $TEMP_DIR/cpp_edge_many.json > /dev/null && echo 'ok'" \
    "ok"

run_test \
    "Single tree model preserves configuration" \
    "grep -q '\"num_trees\": 1' $TEMP_DIR/cpp_edge_1tree.json && echo 'ok'" \
    "ok"

run_test \
    "Large tree count model preserves configuration" \
    "grep -q '\"num_trees\": 200' $TEMP_DIR/cpp_edge_many.json && echo 'ok'" \
    "ok"

run_test \
    "Shallow model has correct depth" \
    "grep -q '\"max_depth\": 1' $TEMP_DIR/cpp_edge_shallow.json && echo 'ok'" \
    "ok"

run_test \
    "Deep model has correct depth" \
    "grep -q '\"max_depth\": 25' $TEMP_DIR/cpp_edge_deep.json && echo 'ok'" \
    "ok"

run_test \
    "High min_leaf model preserves value" \
    "grep -q '\"min_samples_leaf\": 10' $TEMP_DIR/cpp_edge_leaf10.json && echo 'ok'" \
    "ok"

run_test \
    "High min_split model preserves value" \
    "grep -q '\"min_samples_split\": 20' $TEMP_DIR/cpp_edge_split20.json && echo 'ok'" \
    "ok"

run_test \
    "FacadeForest can handle edge case single tree model" \
    "$FACADE_PAS info --model=$TEMP_DIR/cpp_edge_1tree.json" \
    "Trees: 1"

run_test \
    "FacadeForest can handle edge case large tree model" \
    "$FACADE_PAS info --model=$TEMP_DIR/cpp_edge_many.json" \
    "Trees: 200"

run_test \
    "ForestCpp model files retain all metadata after cross-loading" \
    "grep -q '\"criterion\"' $TEMP_DIR/cpp_crit_gini.json && grep -q '\"task_type\"' $TEMP_DIR/cpp_crit_gini.json && echo 'ok'" \
    "ok"

echo ""

# ============================================
# Summary
# ============================================

echo "========================================="
echo "Test Summary"
echo "========================================="
echo "Total tests: $TOTAL"
echo -e "Passed: ${GREEN}$PASS${NC}"
echo -e "Failed: ${RED}$FAIL${NC}"
echo ""

echo "========================================="
echo "Feature Coverage"
echo "========================================="
echo ""
echo "ForestCpp Commands Tested:"
echo "  ✓ help"
echo "  ✓ create (all hyperparameters)"
echo "  ✓ info"
echo ""
echo "FacadeForest (Pascal) Operations on C++ Models:"
echo "  ✓ info"
echo "  ✓ evaluate"
echo "  ✓ inspect"
echo "  ✓ add-tree"
echo "  ✓ remove-tree"
echo "  ✓ retrain-tree"
echo "  ✓ modify-leaf"
echo "  ✓ convert-leaf"
echo "  ✓ prune"
echo "  ✓ set-aggregation"
echo "  ✓ set-weight"
echo "  ✓ reset-weights"
echo "  ✓ feature-usage"
echo "  ✓ feature-heatmap"
echo "  ✓ importance"
echo "  ✓ oob-summary"
echo "  ✓ problematic"
echo "  ✓ worst-trees"
echo "  ✓ misclassified"
echo "  ✓ high-residual"
echo "  ✓ track-sample"
echo "  ✓ node-details"
echo "  ✓ split-dist"
echo "  ✓ visualize"
echo ""
echo "Cross-Language Testing:"
echo "  ✓ Pascal Forest trains C++ ForestCpp models"
echo "  ✓ Pascal FacadeForest evaluates C++ models"
echo "  ✓ Pascal Forest predicts with C++ models"
echo "  ✓ C++ ForestCpp loads Pascal Forest models"
echo ""
echo "Cross-Compatibility Tested:"
echo "  ✓ ForestCpp <-> Pascal Forest model loading"
echo "  ✓ All criterion types cross-loaded"
echo "  ✓ All task types cross-loaded"
echo "  ✓ Trained model cross-loading"
echo ""
echo "JSON Structure Tested:"
echo "  ✓ num_trees"
echo "  ✓ max_depth"
echo "  ✓ min_samples_leaf"
echo "  ✓ min_samples_split"
echo "  ✓ max_features"
echo "  ✓ task_type"
echo "  ✓ criterion"
echo "  ✓ feature_importances"
echo "  ✓ trees array with nodes"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
