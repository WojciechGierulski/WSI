import bnlearn as bn
from pgmpy.factors.discrete import TabularCPD

"""
edge = [('flu', 'fever')]
flu = TabularCPD(values=[[0.95], [0.05]], variable='flu', variable_card=2)
fever = TabularCPD(values=[[0.8, 0.1], [0.2, 0.9]],variable='fever', variable_card=2, evidence=['flu'], evidence_card=[2])

DAG = bn.make_DAG(edge)
model = bn.make_DAG(DAG, CPD=[flu, fever])

bn.inference.fit(model, variables=['flu'], evidence={'fever': 1})"""


edge = [('rain', 'sprinkler'), ('sprinkler', 'wet'), ('rain', 'wet')]
rain = TabularCPD(values=[[0.7], [0.3]], variable='rain', variable_card=2)
sprinkler = TabularCPD(variable_card=2, variable='sprinkler', evidence=['rain'], evidence_card=[2], values=[[0.9, 0.2], [0.1, 0.8]])
wet = TabularCPD(variable_card=2, variable="wet", evidence=['rain', 'sprinkler'], evidence_card=[2], values=[])

DAG = bn.make_DAG(edge)
model = bn.make_DAG(DAG, CPD=[edge, rain, sprinkler, wet])

#bn.inference.fit(model, variables=['flu'], evidence={'fever': 1})