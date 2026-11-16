import Foundation
import KwipMLCore

/// Human-entered lists use the same typed field contracts as single-run settings.
struct WorkflowExperimentDraft:Equatable {
    var values:[String:String]=[:]
    var maximumRuns="8"
    var maximumTotalEpochs="1000"
    private var order=WorkflowControlContracts.experimentFields.map(\.key)
    init(specification:WorkflowExperimentSpecification?=nil) {
        guard let specification else {return}
        order=specification.axes.map(\.field)+order.filter {key in !specification.axes.contains {$0.field == key}}
        maximumRuns=String(specification.budget.maximumRuns)
        maximumTotalEpochs=String(specification.budget.maximumTotalEpochs)
        for axis in specification.axes {
            values[axis.field]=axis.values.map {value in
                switch value {
                case .integer(let number):return String(number)
                case .number(let number):return String(number)
                case .string(let text):return text
                default:return "Invalid value"
                }
            }.joined(separator:", ")
        }
    }
    func specification()throws->WorkflowExperimentSpecification {
        func failure(_ text:String)->DataFlowContractError {.init(text)}
        let fields=WorkflowControlContracts.experimentFields
        guard Set(values.keys).isSubset(of:Set(fields.map(\.key))),
              maximumRuns.count<=8,maximumTotalEpochs.count<=8,
              let runs=Int(maximumRuns),let epochs=Int(maximumTotalEpochs) else {
            throw failure("Enter whole-number run and total epoch limits.")
        }
        let axes=try order.compactMap {key->WorkflowExperimentAxis? in
            guard let text=values[key],!text.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty else {return nil}
            guard text.count<=4096,let field=fields.first(where:{$0.key == key}) else {throw failure("Experiment values are too long or unsupported.")}
            let tokens=text.components(separatedBy:",").map {$0.trimmingCharacters(in:.whitespacesAndNewlines)}
            guard tokens.count<=32,!tokens.contains("") else {throw failure("\(field.title): enter up to 32 comma-separated values without empty entries.")}
            let parsed=try tokens.map {token->JSONValue in
                let value:JSONValue
                switch field.kind {
                case .integer:
                    guard let number=Int64(token) else {throw failure("\(field.title) requires whole numbers.")}
                    value = .integer(number)
                case .positiveDecimal,.nonnegativeDecimal:
                    guard let number=Double(token),number.isFinite else {throw failure("\(field.title) requires finite numbers.")}
                    value = .number(number)
                case .choice:value = .string(token)
                }
                guard field.accepts(value) else {throw failure("\(field.title) has a value outside its supported range.")}
                return value
            }
            return .init(field:key,values:parsed)
        }
        let result=WorkflowExperimentSpecification(axes:axes,budget:.init(maximumRuns:runs,maximumTotalEpochs:epochs))
        try result.validate();return result
    }
    func summary(baseEpochs:Int)throws->(runs:Int,epochs:Int) {
        let spec=try specification()
        let runs=spec.axes.reduce(1) {$0*$1.values.count}
        let epochAxis=spec.axes.first {$0.field == "epochs"}
        let epochs:Int
        if let axis=epochAxis {
            epochs=axis.values.reduce(0) {sum,value in
                if case .integer(let number)=value {return sum+Int(number)};return sum
            }*(runs/axis.values.count)
        } else {
            guard (1...10000).contains(baseEpochs) else {throw DataFlowContractError("The ordinary epoch setting is invalid.")}
            epochs=baseEpochs*runs
        }
        guard epochs<=spec.budget.maximumTotalEpochs else {throw DataFlowContractError("Combinations exceed the total epoch budget. Reduce values or increase the limit.")}
        return(runs,epochs)
    }
}
