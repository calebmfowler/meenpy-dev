from meenpy.mechanics import Force, FBD
from numpy import array

testForce = Force(
    array([1, 1, 1]),
    array([1, 1, 1])
)
print(testForce)

testReaction1 = Force(
    array([1, 1, 1]),
    array([0, 0, 0])
)
testReaction2 = Force(
    array([1, 1, 1]),
    array([1, 0, 0])
)
testFBD = FBD([testForce], [testReaction1, testReaction2])

print(testFBD)

testFBD.solveReactions()