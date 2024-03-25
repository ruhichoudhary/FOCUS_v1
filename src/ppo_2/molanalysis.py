from tdc import Evaluator
evaluator = Evaluator(name = 'Diversity')
evaluator(['CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1', \
            'C[C@@H]1CCc2c(sc(NC(=O)c3ccco3)c2C(N)=O)C1', \
            'CCNC(=O)c1ccc(NC(=O)N2CC[C@H](C)[C@H](O)C2)c(C)c1', \
            'C[C@@H]1CCN(C(=O)CCCc2ccccc2)C[C@@H]1O'])

evaluator = Evaluator(name = 'Validity')
generated = ['CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1', \
            'CCNC(=O)c1ccc(NC(=O)N2CC[C@H](C)[C@H](O)C2)c(C)c1', \
            'C[C@@H]1CCN(C(=O)CCCc2ccccc2)C[C@@H]1O']
evaluator(generated)

evaluator = Evaluator(name = 'Uniqueness')
generated = ['CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1', \
            'CCNC(=O)c1ccc(NC(=O)N2CC[C@H](C)[C@H](O)C2)c(C)c1', \
            'C[C@@H]1CCN(C(=O)CCCc2ccccc2)C[C@@H]1O']
evaluator(generated)

evaluator = Evaluator(name = 'Novelty')
generated = ['CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1', \
            'CCNC(=O)c1ccc(NC(=O)N2CC[C@H](C)[C@H](O)C2)c(C)c1', \
            'C[C@@H]1CCN(C(=O)CCCc2ccccc2)C[C@@H]1O']
training = ['CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1', \
            'C[C@@H]1CCc2c(sc(NC(=O)c3ccco3)c2C(N)=O)C1']
evaluator(generated, training)

evaluator = Evaluator(name = 'FCD_Distance')
generated = ['CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1', \
            'CCNC(=O)c1ccc(NC(=O)N2CC[C@H](C)[C@H](O)C2)c(C)c1', \
            'C[C@@H]1CCN(C(=O)CCCc2ccccc2)C[C@@H]1O']
training = ['CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1', \
            'C[C@@H]1CCc2c(sc(NC(=O)c3ccco3)c2C(N)=O)C1']
evaluator(generated, training)