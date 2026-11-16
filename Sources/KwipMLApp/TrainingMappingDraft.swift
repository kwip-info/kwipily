import Foundation
import KwipMLCore

/// Row identity survives ordering changes, so retained field callbacks cannot
/// accidentally edit the row that later occupies the same array index.
struct TrainingMappingDraft:Equatable {
    struct Row:Identifiable,Equatable {
        let id:UUID
        var name:String
    }
    private(set) var rows:[Row]
    var target:String
    init(features:[String],target:String) {
        rows=features.map {Row(id:UUID(),name:$0)}
        self.target=target
    }
    var features:[String] {rows.map(\.name)}
    mutating func setName(_ name:String,for id:UUID) {
        guard let index=rows.firstIndex(where:{$0.id==id}) else {return}
        rows[index].name=name
    }
    mutating func add() {
        guard rows.count<TrainingColumnMapping.maximumColumns else {return}
        rows.append(Row(id:UUID(),name:""))
    }
    mutating func remove(_ id:UUID) {
        guard rows.count>1 else {return}
        rows.removeAll {$0.id==id}
    }
    mutating func move(_ id:UUID,by offset:Int) {
        guard [-1,1].contains(offset),let index=rows.firstIndex(where:{$0.id==id}),
              rows.indices.contains(index+offset) else {return}
        rows.swapAt(index,index+offset)
    }
}
