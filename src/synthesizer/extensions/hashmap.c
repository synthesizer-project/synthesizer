/******************************************************************************
 * A C module defining a hash map data structure.
 *****************************************************************************/
#include "hashmap.h"

/**
 * @brief Hash function for the hash map.
 *
 * @param key The key to hash.
 * @param ndim The dimension of the key.
 */
unsigned long hash(const IndexKey key, const int ndim) {

  /* Initialise the hash. */
  unsigned long hash = 5381;
  int i;

  /* Hash the key. */
  for (i = 0; i < ndim; i++) {
    hash = ((hash << 5) + hash) + key.indices[i];
  }
  return hash;
}

/**
 * @brief Create a key for the hash map.
 *
 * @param particle_index The index of the particle.
 * @param grid_indices The indices of the grid cell.
 * @param ndim The dimension of the key.
 */
IndexKey create_key(const int particle_index, const int *grid_indices,
                    const int ndim) {

  /* Create the key. */
  IndexKey key;

  /* Allocate the index arrays. */
  key.indices = (int *)malloc(ndim * sizeof(int));
  key.grid_indices = (int *)malloc((ndim - 1) * sizeof(int));

  /* Store the grid indices in the front end of the whole index array. */
  for (int i = 0; i < ndim - 1; i++) {
    key.grid_indices[i] = grid_indices[i];
    key.indices[i] = grid_indices[i];
  }

  /* Store the particle index. */
  key.particle_index = particle_index;
  key.indices[ndim - 1] = particle_index;

  return key;
}

/**
 * @brief Free the memory associated with a key.
 *
 * @param key The key to free.
 */
void free_key(IndexKey key) {
  free(key.indices);
  free(key.grid_indices);
}

/**
 * @brief Compare two keys.
 *
 * @param key1 The first key.
 * @param key2 The second key.
 * @param ndim The dimension of the keys.
 */
int compare_keys(const IndexKey key1, const IndexKey key2, const int ndim) {

  /* Compare the keys. */
  for (int i = 0; i < ndim; i++) {
    if (key1.indices[i] != key2.indices[i]) {
      return 0;
    }
  }
  return 1;
}

/**
 * @brief Create a hash map.
 *
 * @param ndim The dimension of the key.
 */
HashMap *create_hash_map(const int ndim) {

  /* Create the hash map. */
  HashMap *hash_map = (HashMap *)malloc(sizeof(HashMap));
  hash_map->buckets = (Node **)malloc(INITIAL_SIZE * sizeof(Node *));
  hash_map->size = INITIAL_SIZE;
  hash_map->count = 0;
  hash_map->dimension = ndim;

  /* Initialise the buckets. */
  for (int i = 0; i < INITIAL_SIZE; i++) {
    hash_map->buckets[i] = NULL;
  }

  /* Allocate the initial node pool. */
  hash_map->node_pool = (Node *)malloc(NODE_POOL_SIZE * sizeof(Node));
  hash_map->node_pool_count = NODE_POOL_SIZE;
  hash_map->node_pool_used = 0;

  return hash_map;
}

/**
 * @brief Free the memory associated with a hash map.
 *
 * @param hash_map The hash map to free.
 */
void free_hash_map(HashMap *hash_map) {

  /* Free the buckets. */
  for (int i = 0; i < hash_map->size; i++) {
    Node *node = hash_map->buckets[i];
    while (node != NULL) {
      Node *next = node->next;
      free_key(node->key);
      free(node);
      node = next;
    }
  }

  /* Free the node pool. */
  free(hash_map->node_pool);

  /* Free the hash map. */
  free(hash_map->buckets);
  free(hash_map);
}

/**
 * @brief Resize the hash map.
 *
 * @param hash_map The hash map.
 */
void resize(HashMap *map) {

  /* Get the new size. */
  int new_size = map->size * 2;

#ifdef WITH_DEBUGGING_CHECKS
  printf("Resizing hash map from %d to %d\n", map->size, new_size);
#endif

  /* Create the new buckets. */
  Node **new_buckets = malloc(sizeof(Node *) * new_size);

  /* Initialise the new buckets. */
  for (int i = 0; i < new_size; i++) {
    new_buckets[i] = NULL;
  }

  /* Rehash the elements. */
  for (int i = 0; i < map->size; i++) {
    Node *node = map->buckets[i];
    while (node) {
      unsigned long index = hash(node->key, map->dimension) % new_size;
      Node *next = node->next;
      node->next = new_buckets[index];
      new_buckets[index] = node;
      node = next;
    }
  }

  /* Free the old buckets. */
  free(map->buckets);

  /* Update the hash map. */
  map->buckets = new_buckets;
  map->size = new_size;
}

/**
 * @brief Get a value from the hash map.
 *
 * @param hash_map The hash map.
 * @param key The key.
 */
double get(const HashMap *hash_map, const IndexKey key) {

  /* Hash the key. */
  unsigned long h = hash(key, hash_map->dimension) % hash_map->size;

  /* Search for the key. */
  Node *node = hash_map->buckets[h];
  while (node != NULL) {
    if (compare_keys(node->key, key, hash_map->dimension)) {
      return node->value;
    }
    node = node->next;
  }

  /* Return a value to signal that we've not found the key. */
  return -1;
}

/**
 * @brief Insert a key-value pair into the hash map.
 *
 * @param hash_map The hash map.
 * @param key The key.
 * @param value The value.
 */
void insert(HashMap *hash_map, const IndexKey key, const double value) {

  /* Hash the key. */
  unsigned long h = hash(key, hash_map->dimension) % hash_map->size;

  /* Search for the key. */
  Node *node = hash_map->buckets[h];
  while (node != NULL) {

    /* If we find the key update it. */
    if (compare_keys(node->key, key, hash_map->dimension)) {
      node->value += value;
      return;
    }
    node = node->next;
  }

  /* Ok, if we got here it doesn't exist, lets make a new node. */
  node = &hash_map->node_pool[hash_map->node_pool_used++];
  node->key = key;
  node->value = value;

  /* Insert the node. */
  node->next = hash_map->buckets[h];
  hash_map->buckets[h] = node;
  hash_map->count++;

  /* Check if the hash map needs to be resized. */
  if ((double)hash_map->count / hash_map->size > LOAD_FACTOR_THRESHOLD) {

    /* Resize the hash map. */
    resize(hash_map);
  }

  /* Have we exceeded the node pool? */
  else if (hash_map->node_pool_used == hash_map->node_pool_count) {

    /* Allocate a new node pool. */
    hash_map->node_pool = (Node *)malloc(NODE_POOL_SIZE * sizeof(Node));
    hash_map->node_pool_count += NODE_POOL_SIZE;
    hash_map->node_pool_used = 0;
  }
}

/**
 * @brief Delete a key from the hash map.
 *
 * @param hash_map The hash map.
 * @param key The key.
 */
void delete_key(HashMap *hash_map, IndexKey key) {

  /* Hash the key. */
  unsigned long h = hash(key, hash_map->dimension) % hash_map->size;

  /* Search for the key. */
  Node *node = hash_map->buckets[h];
  Node *prev = NULL;
  while (node != NULL) {

    /* If we find the key delete it. */
    if (compare_keys(node->key, key, hash_map->dimension)) {

      /* Update the linked list. */
      if (prev == NULL) {
        hash_map->buckets[h] = node->next;
      } else {
        prev->next = node->next;
      }

      /* Free the memory. */
      free_key(node->key);
      free(node);
      hash_map->count--;
      return;
    }

    /* Move to the next node. */
    prev = node;
    node = node->next;
  }
  free_key(key);
}
