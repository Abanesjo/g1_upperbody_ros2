/****************************************************************

  Hand-authored header for the rosgraph_msgs/msg/Clock message.
  Modeled on the CycloneDDS idlc-generated unitree idl headers
  (e.g. unitree/idl/ros2/Time_.hpp, PoseStamped_.hpp).

  The idlc C++ generator plugin (libcycloneddsidlcxx.so) was not
  available in any toolchain on this system (host ROS jazzy ships
  only the C generator; Ubuntu 22.04 jammy's cyclonedds-tools is
  v0.8.2, mismatching the vendored ddscxx v0.10.2), so this header
  is written by hand following the exact generated-header pattern.

  XTypes type_map/type_info blobs are intentionally omitted and the
  Traits specialize to size 0 / nullptr so CycloneDDS treats the
  type as having no type-discovery information. Matching against
  ROS2 subscribers (rosgraph_msgs::msg::dds_::Clock_) is by type
  name + positional CDR layout, which for @final structs is
  identical on both sides (clock -> Time_ -> {int32 sec, uint32
  nanosec}).

  Type name: rosgraph_msgs::msg::dds_::Clock_
  DDS topic : rt/clock  (ROS2 maps /clock -> rt/clock)

*****************************************************************/
#ifndef DDSCXX_UNITREE_IDL_ROS2_CLOCK__HPP
#define DDSCXX_UNITREE_IDL_ROS2_CLOCK__HPP

#include "unitree/idl/ros2/Time_.hpp"


namespace rosgraph_msgs
{
namespace msg
{
namespace dds_
{
class Clock_
{
private:
 ::builtin_interfaces::msg::dds_::Time_ clock_;

public:
  Clock_() = default;

  explicit Clock_(
    const ::builtin_interfaces::msg::dds_::Time_& clock) :
    clock_(clock) { }

  const ::builtin_interfaces::msg::dds_::Time_& clock() const { return this->clock_; }
  ::builtin_interfaces::msg::dds_::Time_& clock() { return this->clock_; }
  void clock(const ::builtin_interfaces::msg::dds_::Time_& _val_) { this->clock_ = _val_; }
  void clock(::builtin_interfaces::msg::dds_::Time_&& _val_) { this->clock_ = _val_; }

  bool operator==(const Clock_& _other_) const
  {
    (void) _other_;
    return clock_ == _other_.clock_;
  }

  bool operator!=(const Clock_& _other_) const
  {
    return !(*this == _other_);
  }

};

}

}

}

#include "dds/topic/TopicTraits.hpp"
#include "org/eclipse/cyclonedds/topic/datatopic.hpp"

namespace org {
namespace eclipse {
namespace cyclonedds {
namespace topic {

template <> constexpr const char* TopicTraits<::rosgraph_msgs::msg::dds_::Clock_>::getTypeName()
{
  return "rosgraph_msgs::msg::dds_::Clock_";
}

template <> constexpr bool TopicTraits<::rosgraph_msgs::msg::dds_::Clock_>::isSelfContained()
{
  return true;
}

template <> constexpr bool TopicTraits<::rosgraph_msgs::msg::dds_::Clock_>::isKeyless()
{
  return true;
}

#ifdef DDSCXX_HAS_TYPE_DISCOVERY
// No XTypes type-map/type-info blobs: specialize to size 0 / nullptr so the
// base Trait's get_type_info / get_type_map return null gracefully and
// CycloneDDS falls back to type-name + CDR matching.
template <> constexpr unsigned int TopicTraits<::rosgraph_msgs::msg::dds_::Clock_>::type_map_blob_sz() { return 0u; }
template <> inline const uint8_t * TopicTraits<::rosgraph_msgs::msg::dds_::Clock_>::type_map_blob() { return nullptr; }
template <> constexpr unsigned int TopicTraits<::rosgraph_msgs::msg::dds_::Clock_>::type_info_blob_sz() { return 0u; }
template <> inline const uint8_t * TopicTraits<::rosgraph_msgs::msg::dds_::Clock_>::type_info_blob() { return nullptr; }
#endif //DDSCXX_HAS_TYPE_DISCOVERY

} //namespace topic
} //namespace cyclonedds
} //namespace eclipse
} //namespace org

namespace dds {
namespace topic {

template <>
struct topic_type_name<::rosgraph_msgs::msg::dds_::Clock_>
{
    static std::string value()
    {
      return org::eclipse::cyclonedds::topic::TopicTraits<::rosgraph_msgs::msg::dds_::Clock_>::getTypeName();
    }
};

}
}

REGISTER_TOPIC_TYPE(::rosgraph_msgs::msg::dds_::Clock_)

namespace org{
namespace eclipse{
namespace cyclonedds{
namespace core{
namespace cdr{

template<>
propvec &get_type_props<::rosgraph_msgs::msg::dds_::Clock_>();

template<typename T, std::enable_if_t<std::is_base_of<cdr_stream, T>::value, bool> = true >
bool write(T& streamer, const ::rosgraph_msgs::msg::dds_::Clock_& instance, entity_properties_t *props) {
  (void)instance;
  if (!streamer.start_struct(*props))
    return false;
  auto prop = streamer.first_entity(props);
  while (prop) {
    switch (prop->m_id) {
      case 0:
      if (!streamer.start_member(*prop))
        return false;
      if (!write(streamer, instance.clock(), prop))
        return false;
      if (!streamer.finish_member(*prop))
        return false;
      break;
    }
    prop = streamer.next_entity(prop);
  }
  return streamer.finish_struct(*props);
}

template<typename S, std::enable_if_t<std::is_base_of<cdr_stream, S>::value, bool> = true >
bool write(S& str, const ::rosgraph_msgs::msg::dds_::Clock_& instance, bool as_key) {
  auto &props = get_type_props<::rosgraph_msgs::msg::dds_::Clock_>();
  str.set_mode(cdr_stream::stream_mode::write, as_key);
  return write(str, instance, props.data()); 
}

template<typename T, std::enable_if_t<std::is_base_of<cdr_stream, T>::value, bool> = true >
bool read(T& streamer, ::rosgraph_msgs::msg::dds_::Clock_& instance, entity_properties_t *props) {
  (void)instance;
  if (!streamer.start_struct(*props))
    return false;
  auto prop = streamer.first_entity(props);
  while (prop) {
    switch (prop->m_id) {
      case 0:
      if (!streamer.start_member(*prop))
        return false;
      if (!read(streamer, instance.clock(), prop))
        return false;
      if (!streamer.finish_member(*prop))
        return false;
      break;
    }
    prop = streamer.next_entity(prop);
  }
  return streamer.finish_struct(*props);
}

template<typename S, std::enable_if_t<std::is_base_of<cdr_stream, S>::value, bool> = true >
bool read(S& str, ::rosgraph_msgs::msg::dds_::Clock_& instance, bool as_key) {
  auto &props = get_type_props<::rosgraph_msgs::msg::dds_::Clock_>();
  str.set_mode(cdr_stream::stream_mode::read, as_key);
  return read(str, instance, props.data()); 
}

template<typename T, std::enable_if_t<std::is_base_of<cdr_stream, T>::value, bool> = true >
bool move(T& streamer, const ::rosgraph_msgs::msg::dds_::Clock_& instance, entity_properties_t *props) {
  (void)instance;
  if (!streamer.start_struct(*props))
    return false;
  auto prop = streamer.first_entity(props);
  while (prop) {
    switch (prop->m_id) {
      case 0:
      if (!streamer.start_member(*prop))
        return false;
      if (!move(streamer, instance.clock(), prop))
        return false;
      if (!streamer.finish_member(*prop))
        return false;
      break;
    }
    prop = streamer.next_entity(prop);
  }
  return streamer.finish_struct(*props);
}

template<typename S, std::enable_if_t<std::is_base_of<cdr_stream, S>::value, bool> = true >
bool move(S& str, const ::rosgraph_msgs::msg::dds_::Clock_& instance, bool as_key) {
  auto &props = get_type_props<::rosgraph_msgs::msg::dds_::Clock_>();
  str.set_mode(cdr_stream::stream_mode::move, as_key);
  return move(str, instance, props.data()); 
}

template<typename T, std::enable_if_t<std::is_base_of<cdr_stream, T>::value, bool> = true >
bool max(T& streamer, const ::rosgraph_msgs::msg::dds_::Clock_& instance, entity_properties_t *props) {
  (void)instance;
  if (!streamer.start_struct(*props))
    return false;
  auto prop = streamer.first_entity(props);
  while (prop) {
    switch (prop->m_id) {
      case 0:
      if (!streamer.start_member(*prop))
        return false;
      if (!max(streamer, instance.clock(), prop))
        return false;
      if (!streamer.finish_member(*prop))
        return false;
      break;
    }
    prop = streamer.next_entity(prop);
  }
  return streamer.finish_struct(*props);
}

template<typename S, std::enable_if_t<std::is_base_of<cdr_stream, S>::value, bool> = true >
bool max(S& str, const ::rosgraph_msgs::msg::dds_::Clock_& instance, bool as_key) {
  auto &props = get_type_props<::rosgraph_msgs::msg::dds_::Clock_>();
  str.set_mode(cdr_stream::stream_mode::max, as_key);
  return max(str, instance, props.data()); 
}

} //namespace cdr
} //namespace core
} //namespace cyclonedds
} //namespace eclipse
} //namespace org

#endif // DDSCXX_UNITREE_IDL_ROS2_CLOCK__HPP
