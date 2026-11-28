import json
import dataclasses

@dataclasses.dataclass
class VowelChangeRecord:
    vw0:str
    des_chg:str
    score:float
    len:int
    vw1:dict[str, list[tuple[str, str]]]
    comments: str = ''
    vw1_ordered : dict[str, float] = None
    word_lists: list[list[tuple[str, str]]] = None

    def __post_init__(self):
        # sanity
        self.len = sum([len(v) for v in self.vw1.values()])
        vw1_items = list(self.vw1.items())
        vw1_items.sort(key=lambda t: len(t[1])) # sort by len of wordlist
        self.vw1_ordered = {k:len(v)/self.len for k,v in vw1_items}
        self.word_lists = [v for _, v in vw1_items]

    @staticmethod
    def from_dict(d: dict) -> 'VowelChangeRecord':
        return VowelChangeRecord(**{k:v for k,v in d.items() if k in VowelChangeRecord.__annotations__})

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def __str__(self) -> str:
        return str({k:v for k,v in self.to_dict().items() if 
                    k in ['vw0', 'des_chg', 'vw1_ordered']})
    def __repr__(self) -> str:
        return str(self)

    def outputs(self) -> dict:
        return self.vw1_ordered

    def word_list(self, index:int) -> list[tuple[str, str]]:
        return self.word_lists[index]


def vowel_change_rec_list_to_json(wcr_list: list[VowelChangeRecord], filename: str):
    with open(filename, 'w') as handle:
        json.dump([rec.to_dict() for rec in wcr_list], handle)
        

if __name__ == "__main__":
    FILENAME = './train_data/work5.split.1.json'

    with open(FILENAME, 'r') as handle:
        data_list = json.load(handle)

    vw_list = [VowelChangeRecord.from_dict(d) for d in data_list]
    vw_chgs = [c for c in vw_list if c.score < 1]

