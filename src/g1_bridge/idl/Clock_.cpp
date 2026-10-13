/****************************************************************

  Hand-authored get_type_props definition for rosgraph_msgs::msg::dds_::Clock_.
  Mirrors the Eclipse CycloneDDS idlc-generated pattern (see
  dependencies/unitree_sdk2/example/helloworld/HelloWorldData.cpp).

  Clock_ wraps a single builtin_interfaces::msg::dds_::Time_ member ("clock"),
  so the propvec is: root + clock member + Time_'s children (sec, nanosec)
  appended via append_struct_contents with depth+1, then finish() wires the
  first_member / next_on_level / parent pointers. Verified against the actual
  idlc-generated propvec of Time_ and PoseStamped_ extracted from the
  precompiled libunitree_sdk2.a (PoseStamped_ header->stamp->sec/nanosec).

*****************************************************************/
#include "Clock_.hpp"

namespace org{
namespace eclipse{
namespace cyclonedds{
namespace core{
namespace cdr{

template<>
propvec &get_type_props<::rosgraph_msgs::msg::dds_::Clock_>() {
  static thread_local std::mutex mtx;
  static thread_local propvec props;
  static thread_local entity_properties_t *props_end = nullptr;
  static thread_local std::atomic_bool initialized {false};
  key_endpoint keylist;
  if (initialized.load(std::memory_order_relaxed)) {
    auto ptr = props.data();
    while (ptr < props_end)
      (ptr++)->is_present = false;
    return props;
  }
  std::lock_guard<std::mutex> lock(mtx);
  if (initialized.load(std::memory_order_relaxed)) {
    auto ptr = props.data();
    while (ptr < props_end)
      (ptr++)->is_present = false;
    return props;
  }
  props.clear();

  props.push_back(entity_properties_t(0, 0, false, bb_unset, extensibility::ext_final));  //root
  props.push_back(entity_properties_t(1, 0, false, bb_unset, extensibility::ext_final, false));  //::clock

  entity_properties_t::append_struct_contents(props, get_type_props<::builtin_interfaces::msg::dds_::Time_>());

  entity_properties_t::finish(props, keylist);
  props_end = props.data() + props.size();
  initialized.store(true, std::memory_order_release);
  return props;
}

} //namespace cdr
} //namespace core
} //namespace cyclonedds
} //namespace eclipse
} //namespace org
