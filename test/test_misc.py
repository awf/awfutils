from awfutils import dict_to_simple_namespace


def test_dict_to_simple_namespace():
    d = dict(
        a=3,
        b=dict(
            ba=[3, 4],
            bb="fred",
            bc=(3, 4),
            bd=dict(
                bda=1,
                bdb=2,
            ),
        ),
    )
    ns = dict_to_simple_namespace(d)
    assert ns.a == 3
    assert ns.b.ba == [3, 4]
    assert ns.b.bb == "fred"
    assert ns.b.bc == (3, 4)
    assert ns.b.bd.bda == 1
    assert ns.b.bd.bdb == 2
