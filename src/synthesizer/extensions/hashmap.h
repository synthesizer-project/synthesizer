/******************************************************************************
 * A C module defining a hash map data structure.
 *****************************************************************************/
#ifndef HASHMAP_H_
#define HASHMAP_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Initial size of the hash map
#define INITIAL_SIZE 50
#define LOAD_FACTOR_THRESHOLD 0.75
#define NODE_POOL_SIZE 100

/*! @brief A key for the hash map. */
typedef struct {

  /* Array of indices representing the key */
  int *indices;

  /* Particle index (can be 0). */
  int particle_index;

  /* Indices of the grid */
  int *grid_indices;

} IndexKey;

/*! @brief A node of the hash map containing the data. */
typedef struct Node {

  /* The key of this node. */
  IndexKey key;

  /* The value of this node. */
  double value;

  /* Pointer to the next node in the linked list. */
  struct Node *next;

} Node;

/*! @brief A hash map data structure. */
typedef struct HashMap {

  /* Array of buckets. */
  Node **buckets;

  /* The size of the hash map. */
  int size;

  /* The number of elements in the hash map. */
  int count;

  /* The dimension of the key. */
  int dimension;

  /* The pool of nodes (to avoid multiple small mallocs). */
  Node *node_pool;

  /* The number of nodes in the pool. */
  int node_pool_count;

  /* The number of used nodes in the pool. */
  int node_pool_used;

} HashMap;

/* Prototypes. */
unsigned long hash(const IndexKey key, const int ndim);
IndexKey create_key(const int particle_index, const int *grid_indices,
                    const int ndim);
void free_key(IndexKey key);
int compare_keys(const IndexKey key1, const IndexKey key2, const int ndim);
HashMap *create_hash_map(const int ndim);
void free_hash_map(HashMap *hash_map);
void insert(HashMap *hash_map, const IndexKey key, const double value);
double get(const HashMap *hash_map, const IndexKey key);
void delete_key(HashMap *hash_map, IndexKey key);
void resize(HashMap *hash_map);

#endif // HASHMAP_H_
