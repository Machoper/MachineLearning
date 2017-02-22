#Yixuan Qian
from __future__ import print_function
import json
import random
import time
import math

start_time = time.time()


def load_data(filename):
    data = []
    for line in open(filename):
        data.append(json.loads(line))
    return data


def get_cuisine_probs(data):
    cuisine_count = {}

    # Calculate number of times each ingredient and cuisine appears in the whole dataset
    for recipe in data:
        cuisine = recipe['cuisine']
        cuisine_count[cuisine] = cuisine_count.get(cuisine, 0) + 1

    # Divide by length of dataset
    cuisine_probs = {cuisine : cuisine_count[cuisine]/float(len(data)) for cuisine in cuisine_count}

    return cuisine_probs


def get_ingredient_prob_given_cuisine(data, all_cuisines, all_ingredients):
    # Place a prior over each ingredient given the cuisine
    ingredient_prob_given_cuisine = {cuisine : {ingredient : 0.15 for ingredient in all_ingredients} for cuisine in all_cuisines}

    # Calculate number of times each ingredient occurs for a given cuisine
    for recipe in data:
        cuisine = recipe['cuisine']
        ingredients = recipe['ingredients']
        for ingredient in ingredients:
            ingredient_prob_given_cuisine[cuisine][ingredient] += 1

    # Divide by total number of times the cuisine appears (plus the prior distribution)
    for cuisine in ingredient_prob_given_cuisine:
        probs = ingredient_prob_given_cuisine[cuisine]
        total = float(sum(probs[ingredient] for ingredient in probs))
        ingredient_prob_given_cuisine[cuisine] = {ingredient : probs[ingredient]/total for ingredient in probs}

    return ingredient_prob_given_cuisine


def get_max_cuisine(ingredient_list, cuisine_probs, ingredient_prob_given_cuisine):
    max_prob = 0
    best_cuisine = None

    # Iterate over every cuisine
    for cuisine in cuisine_probs:
        # Set probability to p(cuisine)
        prob = cuisine_probs[cuisine]
        # Multiply by p(ingredient | cuisine) for each ingredient in ingredient_list
        for ingredient in ingredient_list:
            # Here we say .get(ingredient, 1) instead of .get(ingredient, 0)
            # which causes it to ignore ingredients that have never been seen before
            prob *= ingredient_prob_given_cuisine[cuisine].get(ingredient, 1)
        # Check if new prob is higher
        if prob > max_prob:
            max_prob = prob
            best_cuisine = cuisine
    return best_cuisine


def test_classifier(data, k):
    # Split data into training and test set
    slice = int(math.ceil(len(data)/float(6)))
    test_data = data[k*slice:k*slice+slice]
    train_data = [j for j in data if j not in test_data]
    # Train the classifier
    cuisine_probs = get_cuisine_probs(train_data)
    # Get list of all cuisines and all ingredients
    all_cuisines = [cuisine for cuisine in cuisine_probs]
    all_ingredients = set()
    for recipe in train_data:
        for ingredient in recipe['ingredients']:
            all_ingredients.add(ingredient)
    ingredient_prob_given_cuisine = get_ingredient_prob_given_cuisine(train_data, all_cuisines, all_ingredients)

    # Test the classifier
    results = []
    for recipe in test_data:
        cuisine = get_max_cuisine(recipe['ingredients'], cuisine_probs, ingredient_prob_given_cuisine)
        results.append((cuisine, recipe['cuisine']))

    return results


def eval_classifier(results):
    return sum(guessed_cuisine == true_cuisine for (guessed_cuisine, true_cuisine) in results)/float(len(results))

sum_error = 0
dataset = load_data('training.json')
random.shuffle(dataset)
for k in range(0, 6):
    re = test_classifier(dataset, k)
    error = 1 - eval_classifier(re)
    print(k+1, ' trial: ', error)
    sum_error += error

print()
print('Average generalization error: ', sum_error/6)
print("Total time: %s seconds " % (time.time() - start_time))

