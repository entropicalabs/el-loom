.. _block:

Block
=======

Definition
^^^^^^^^^^

A :class:`~loom.eka.block.Block` is a dataclass that describes a QEC code that is compatible with Loom modules and more particularly with :class:`~loom.eka.eka.Eka`.

Note that at no point a :class:`~loom.eka.block.Block` will represent a full quantum state: they describe a subspace in which one can encode logical information using operations.

It is a general representation of a stabilizer code.

.. _block_example:

Example
^^^^^^^
Let's create a :class:`~loom.eka.block.Block` object that represents a simple repetition code.

We start by defining the :class:`~loom.eka.lattice.Lattice` that sets the geometry of the system. Since the repetition code can be represented on a line, we choose the linear lattice:

.. code-block:: python

  from loom.eka import Lattice

  linear_lattice = Lattice.linear(lattice_size=(3,))

While it is not necessary to create a :class:`~loom.eka.block.Block` object, it is useful to know what the lattice is like for indexing purpose.

Once the lattice is defined, we can create the code stabilizers. To create a bit-flip repetition code, we need two :math:`ZZ` checks.

.. code-block:: python
  
  from loom.eka import Stabilizer

  repetition_stabs = [
      Stabilizer(
          pauli="ZZ", 
          data_qubits=((i, 0), (i+1, 0)),
          ancilla_qubits=((i, 1),)
      )
      for i in range(2)
  ]

We need to make sure that the qubit indices are compatible with the geometry we defined previously.

Then come the logical operators. The bit-flip operators are defined by a single :math:`Z` operator and a :math:`XXX` operator (spanning the whole set of data qubits):

.. code-block:: python

  from loom.eka import PauliOperator

  x_op = PauliOperator(pauli="Z", data_qubits=((0, 0),))
  z_op = PauliOperator(pauli="XXX", data_qubits=((0, 0), (1, 0), (2, 0)))

Finally, we need to specify how to measure the stabilizers we created using :class:`~loom.eka.syndrome_circuit.SyndromeCircuit` objects and mapping them with the stabilizers. Since we are measuring the stabilizers of a repetition code, we only need to define a single circuit blueprint that will be used by both stabilizers:

.. code-block:: python

  from loom.eka import SyndromeCircuit

  zz_circuit = SyndromeCircuit(
      pauli="ZZ",
      name="zz",
      circuit=None,   # This creates a default circuit to measure the given pauli string
  )

  stab_to_circuit = {
      stab.uuid: zz_circuit.uuid for stab in repetition_stabs
  }

Note that this step may be ignored. In that case, a default set of :class:`~loom.eka.syndrome_circuit.SyndromeCircuit` is provided but there is no guarantee when it comes to fault tolerance properties.

Once we have defined all the different components, we can assemble them together and create the :class:`~loom.eka.block.Block` object:

.. code-block:: python
  
  from loom.eka import Block

  my_rep_code = Block(
      code_type="custom",
      unique_label="qubit_1",
      stabilizers=repetition_stabs,
      logical_x_operators=[x_op],
      logical_z_operators=[z_op],
      syndrome_circuits=[zz_circuit],
      stabilizer_to_syndrome_circuit=stab_to_circuit,
  )

This process can be automated and we provide multiple code factories for well-known codes.

Validations
^^^^^^^^^^^

There is a comprehensive set of validations applied at the construction of :class:`~loom.eka.block.Block` to ensure that some properties are satisfied. These checks ensure that a user defining a custom code does not result in an erroneous definition of a QEC code. We use the notation :math:`[\![n, k, d]\!]` for a code that encodes :math:`k` logical qubits in :math:`n` physical qubits and whose distance is :math:`d`. 

The list of validators includes:

- Number of logical operators: the number of logical :math:`X` and :math:`Z` operators is the same. They define the number of logical qubits encoded in the :class:`~loom.eka.block.Block`.

  :math:`|\{\mathcal{L}_{X}\}| = |\{\mathcal{L}_{Z}\}|`

- All physical qubits contained in logical operators are also contained in stabilizers.

- All stabilizers and logical operators are distinct: no object is repeated in the specification of the code.

- Commutation relations of stabilizers: all stabilizers should commute one-to-one.

  :math:`[s_i, s_j] = 0, \forall s_i, s_j \in \{\mathcal{S}\}`

- Commutation relations of logical operators: all logical :math:`X` operators should commute one-to-one, all logical :math:`Z` operators should commute one-to-one.

  :math:`[x_i, x_j] = 0, \forall x_i, x_j \in \{\mathcal{L}_X\}`,
  
  :math:`[z_i, z_j] = 0, \forall z_i, z_j \in \{\mathcal{L}_Z\}`
  
- Commutation of logical operators with stabilizers: all logical operators commute with all stabilizers.

  :math:`[l_i, s_j] = 0, \forall l_i \in \{\mathcal{L}_{X,Z}\}, \forall s_j \in \{\mathcal{S}\}`

- Anti-commutation relations of logical operators: all logical :math:`X` operators anti-commute with exactly one logical :math:`Z` operator (and vice-versa). The non-commuting logical operators share the same index by convention.

  :math:`[x_i, z_j] = 0, \forall x_i, z_j \in \{\mathcal{L}_{X,Z}\}, i\neq j`

  :math:`\{x_i, z_i\} = 0, \forall x_i \in \{\mathcal{L}_X\} \text{ and } z_i \in \{\mathcal{L}_Z\}`

- The number of physical qubits, stabilizers, and logical operators is consistent with the code: for a given code :math:`[\![n, k, d]\!]`, we have :math:`n - k` independent stabilizers acting (non-trivially) on a total of :math:`n` qubits, :math:`k` logical :math:`X` operators, and :math:`k` logical :math:`Z` operators. Since only the number of independent stabilizer matters, we can use an overdefined set.

  :math:`\text{support}(\{\mathcal{S}\}) = n`

  :math:`|\{\mathcal{S}\}| = n-k`

  :math:`|\{\mathcal{L}_X\}| = |\{\mathcal{L}_Z\}| = k`

*Note*: for better performance when it comes to object creation, the validation can be turned off by using ``skip_validation=True`` when instantiating a :class:`~loom.eka.block.Block`.
**It should only be set to false if the block arguments are known to produce a valid object**. We recommend using this keyword only for the pre-defined code factories that we provide.

Assumptions
^^^^^^^^^^^

A block assumes that there exists a stabilizer representation of the QEC code and that :class:`~loom.eka.circuit.Circuit` objects are mapped to these stabilizers. Non-stabilizer codes are not natively supported.
