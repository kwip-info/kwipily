import SwiftUI

struct UnionInformationView:View {
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            LessonTextPanel(title:"Append rows",text:"Union stacks tables vertically. Join pairs columns horizontally. Input order controls row order; duplicates stay in the output. Grips or arrows reorder stable ports without moving their connections. Disconnect a port before removing it; every configured input must be connected to run.")
            LessonTextPanel(title:"Align columns",text:"By name matches exact, case-sensitive names and retains first-seen column order. By position maps the first column to the first column, regardless of names; output names come from the first input that supplies each position. Include all fills absent fields with missing values; Shared only drops fields absent from any input; Require identical stops on different names/counts. Alignment is declared guidance until a run verifies source schemas.")
            LessonTextPanel(title:"Types and origins",text:"Require same types stops on mismatches. Checked numeric widening allows integers and numbers to share a numeric column only when every integer is exactly representable. Text, booleans and image references are never implicitly converted. Use Columns for explicit conversions. The optional identifier stores stable port IDs such as table1, independent of input order or source display names.")
            LessonTextPanel(title:"Boundaries",text:"Union streams into owned storage; it has no 100,000-row cap. Shared cell, scratch and artifact budgets still apply. Union belongs before Partition and fitted preprocessing: partitioned/fitted inputs are refused so isolation evidence cannot be lost. Observation identities are qualified by input port, preserving duplicates and allowing the same source on multiple inputs. Repeated entities or duplicate observations need appropriate group-aware partitioning before model evaluation. Working-data sampling applies to each source; View requests complete upstream data.")
        }
    }
}
